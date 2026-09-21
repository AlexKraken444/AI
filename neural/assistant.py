"""Response orchestration. The neural classifier chooses a curated answer.

Status comments are UI narration, not a model's hidden chain of thought.
"""
import ast
import json
import math
import operator
import re
from functools import lru_cache
from pathlib import Path
from neural.network import Network, features


@lru_cache(maxsize=1)
def assets():
    intents = json.loads(Path(__file__).with_name("intents.json").read_text(encoding="utf-8"))
    return Network(), {i["id"]: i for i in intents}, {
        i["id"]: [features(p) for p in i["patterns"]] for i in intents
    }


def calculate(text):
    expression = re.sub(r"^(посчитай|вычисли|сколько будет|реши)\s*[:=]?\s*", "", text.lower()).strip().rstrip("?=").strip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", ".")
    if not expression or len(expression) > 120 or not re.fullmatch(r"[\d\s.+*/()%\-]+", expression):
        return None
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                 ast.Mod: operator.mod, ast.Pow: operator.pow}

    def visit(node, depth=0):
        if depth > 15:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 20 or abs(left) > 1e6):
                raise ValueError("power too large")
            value = operators[type(node.op)](left, right)
        else:
            raise ValueError("unsupported expression")
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("result too large")
        return value

    try:
        value = visit(ast.parse(expression, mode="eval").body)
        return f"{expression} = {value:g}" if isinstance(value, float) else f"{expression} = {value}"
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return "Не получилось вычислить выражение. Проверь скобки и деление на ноль. Поддерживаются +, −, *, /, //, %, степени до 20 и числа с результатом не больше 10¹⁵ по модулю."


def prepare_reply(messages, personality):
    text = messages[-1]["content"].strip()
    lower = text.lower()
    if re.search(r"(как|инструкц|сдела|созда|напиш).{0,70}(бомб|взрывчат|убить|отравить|фишинг|вирус для|взломать)", lower):
        return "Не помогу с инструкциями, которые могут причинить вред. Могу обсудить защиту, предотвращение опасности или безопасный учебный пример.", "Подбираю безопасный вариант ответа.", "safety"
    result = calculate(text)
    if result is not None:
        return result, "Вычисляю выражение с проверкой допустимых операций.", "calculator"
    if re.fullmatch(r"(повтори|повтори ответ|еще раз|ещё раз)[.!?]*", lower):
        previous = next((m["content"] for m in reversed(messages[:-1]) if m["role"] == "assistant"), None)
        return previous or "Пока нет предыдущего ответа. Задай первый вопрос.", "Проверяю предыдущий ответ в этом диалоге.", "context"
    name_match = re.fullmatch(r"меня зовут\s+([а-яёa-z\-]{2,30})[.!]?", text, re.I)
    if name_match:
        return f"Приятно познакомиться, {name_match[1].capitalize()}! Запомню имя, пока это сообщение доступно в контексте чата.", "Знакомимся чуть ближе.", "context"
    if re.search(r"как меня зовут|помнишь мое имя|помнишь моё имя", lower):
        for message in reversed(messages[:-1]):
            match = re.fullmatch(r"меня зовут\s+([а-яёa-z\-]{2,30})[.!]?", message["content"].strip(), re.I)
            if message["role"] == "user" and match:
                return f"Тебя зовут {match[1].capitalize()}.", "Проверяю имя в доступной истории.", "context"
        return "В доступной истории ты ещё не называл имя. Напиши: «Меня зовут …».", "Проверяю доступную историю чата.", "context"
    net, intents, references = assets()
    label, probability = net.predict(text)
    vector = features(text)
    similarity = max(sum(a * b for a, b in zip(vector, reference)) for reference in references[label])
    if probability < .65 or similarity < .38 or len(text.split()) > 45:
        return ("Эта формулировка пока за пределами моих знаний. Я небольшая модель с подготовленными ответами, поэтому не буду придумывать факты.\n\n"
                "Попробуй спросить об основах Python, нейросетях, API или обучении модели. Ещё умею считать: «Посчитай (12 + 8) * 3».",
                "Проверяю, есть ли достаточно похожая тема в обучающих примерах.", "unknown")
    item = intents[label]
    aside = item["aside"] if personality else "Распознана знакомая тема. Подготавливаю ответ из базы знаний."
    answer = item["answer"]
    if personality and label in {"python", "loop", "function", "training", "learning"}:
        answer += "\n\nМаленький совет: запускай примеры. Код, на который только смотришь, почему-то не учится работать сам."
    return answer, aside, label


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Ожидается JSON-объект.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
        raise ValueError("Отправьте от 1 до 24 сообщений.")
    if type(payload.get("personality", True)) is not bool:
        raise ValueError("Параметр personality должен быть логическим.")
    cleaned = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise ValueError("Некорректная роль сообщения.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 4000:
            raise ValueError("Сообщение должно содержать от 1 до 4000 символов.")
        cleaned.append({"role": item["role"], "content": content})
    if cleaned[-1]["role"] != "user":
        raise ValueError("Последнее сообщение должно быть от пользователя.")
    return cleaned, payload.get("personality", True)
