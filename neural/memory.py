"""Validate per-request browser memory. Never persist it in process-global state."""
import re


def validate_memory(value):
    if value is None:
        return {"enabled": False, "facts": [], "excerpts": []}
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("Invalid memory payload")
    if not value["enabled"]:
        return {"enabled": False, "facts": [], "excerpts": []}
    result = {"enabled": True, "facts": [], "excerpts": []}
    for group, limit in (("facts", 8), ("excerpts", 4)):
        items = value.get(group, [])
        if not isinstance(items, list) or len(items) > limit:
            raise ValueError("Too many memory records")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Invalid memory record")
            fields = ("id", "chatId", "key", "value", "text") if group == "facts" else ("id", "chatId", "title", "text")
            clean = {}
            for field in fields:
                content = item.get(field)
                if not isinstance(content, str) or not content.strip() or len(content) > (600 if field == "text" else 400):
                    raise ValueError("Invalid memory field")
                clean[field] = content
            result[group].append(clean)
    saved = value.get("saved", [])
    if not isinstance(saved, list) or len(saved) > 8 or any(not isinstance(x, str) or len(x) > 400 for x in saved):
        raise ValueError("Invalid saved record list")
    result["saved"] = [record for record in result["facts"] if record["id"] in saved]
    return result


def memory_reply(text, memory):
    """Grounded personal-fact lookup; this is software, not claimed neural reasoning."""
    query = text.lower().replace("ё", "е")
    patterns = {
        "name": r"как меня зовут|мое имя|помнишь.{0,12}имя",
        "city": r"где я живу|мой город|в каком городе я",
        "study": r"что я (?:учу|изучаю)|какой язык я (?:учу|изучаю)",
        "project": r"как называется мой проект|название моего проекта",
        "goal": r"какая у меня цель|какая моя цель",
        "likes": r"что мне нравится|что я люблю",
        "dislikes": r"что я не люблю|что мне не нравится",
    }
    for key, pattern in patterns.items():
        if re.search(pattern, query):
            record = next((f for f in memory["facts"] if f["key"] == key), None)
            if record:
                formats = {"name": "Тебя зовут {}.", "city": "Ты говорил, что живёшь в {}.", "study": "Ты изучаешь {}.", "project": "Твой проект называется {}.", "goal": "Твоя сохранённая цель: {}.", "likes": "Тебе нравится {}.", "dislikes": "Ты говорил, что не любишь {}."}
                return formats[key].format(record["value"].rstrip(".!")), [record]
    if re.search(r"что ты (?:обо мне |про меня )?(?:помнишь|знаешь)|что (?:запомнил|сохранено в памяти)", query):
        facts = memory["facts"]
        answer = "В переданной мне памяти:\n\n" + "\n".join("• " + f["text"] for f in facts) if facts else "В переданной мне памяти пока нет фактов о тебе. Можно написать «Запомни: …»."
        return answer, facts
    if re.search(r"(?:что|о чем).{0,35}(?:обсуждали|говорили)|прошл.{0,12}чат|предыдущ.{0,12}чат", query):
        excerpts = memory["excerpts"]
        if excerpts:
            return "Нашёл твои сообщения из прошлых диалогов:\n\n" + "\n\n".join(f'«{item["title"]}»: «{item["text"]}»' for item in excerpts), excerpts
        return "Подходящих сообщений из прошлых чатов в переданном контексте нет. Проверь, включена ли память и сохранены ли эти чаты в этом браузере.", []
    return None


def memory_context(memory, tokenizer=None, messages=None):
    """Bounded user-data context, not an instruction/system prompt."""
    if not memory["enabled"]:
        return "", []
    candidates = memory["facts"][:3] + memory["excerpts"][:1]
    selected, lines = [], []
    budget = 80
    if tokenizer is not None and messages:
        latest_length = len(tokenizer.encode(messages[-1]["content"]))
        if latest_length > 252:
            latest_length = 252
        budget = min(80, max(0, (252 - latest_length) // 2 - 2))
    for item in candidates:
        line = "Меня зовут " + item["value"] + "." if item.get("key") == "name" else item["text"]
        proposed = "\n".join(lines + [line])
        if len(proposed) > 1000 or tokenizer is not None and len(tokenizer.encode(proposed)) > budget:
            continue
        lines.append(line)
        selected.append(item)
    return "\n".join(lines), selected
