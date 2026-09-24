"""Request pipeline: grounded tools/memory + autoregressive text generation."""
from functools import lru_cache
from pathlib import Path
import re
from neural.assistant import prepare_reply
from neural.memory import memory_reply, memory_context


@lru_cache(maxsize=3)
def language_model(mode="context4"):
    from neural.transformer import Transformer
    directories = {"context": "checkpoint", "context3": "checkpoint3", "context4": "checkpoint4"}
    return Transformer(Path(__file__).parent / directories[mode])


def source_event(sources):
    return {"type": "sources", "items": [
        {"id": item["id"], "chatId": item["chatId"], "title": item.get("title", "Сохранённый факт"), "text": item["text"]}
        for item in sources
    ]}


def text_events(answer, route, model="Kraken Context 4"):
    for offset in range(0, len(answer), 48):
        yield {"type": "token", "text": answer[offset:offset + 48]}
    yield {"type": "done", "model": model, "route": route, "finish_reason": "end"}


def reply_events(messages, personality, memory, mode="context4"):
    text = messages[-1]["content"].strip()
    model_name = {"context4": "Kraken Context 4", "context3": "Kraken Context 3", "context": "Kraken Context 2", "reference": "Kraken Mini"}[mode]
    yield {"type": "status", "text": "Проверяю доступный контекст и сохранённую память."}
    recalled = memory_reply(text, memory)
    if recalled:
        answer, sources = recalled
        yield {"type": "status", "text": f"Нашёл записей в памяти: {len(sources)}. Сверяю ответ с источниками."}
        yield source_event(sources)
        yield from text_events(answer, "memory", model_name)
        return
    if re.match(r"^запомни\s*[:—-]?\s+", text, re.I):
        if not memory["enabled"]:
            answer = "Память выключена. Включи её в разделе «Память», чтобы сохранять факты между чатами."
        elif not memory.get("saved"):
            answer = "Не вижу сохранённой записи. Проверь раздел «Память»; пароли и ключи автоматически не сохраняются."
        else:
            answer = "Запись сохранена в памяти этого браузера. Её можно проверить или удалить в разделе «Память»."
            yield source_event(memory["saved"])
        yield from text_events(answer, "memory_saved", model_name)
        return
    if memory.get("saved"):
        saved = memory["saved"]
        yield source_event(saved)
        yield from text_events("Запомнил в этом браузере:\n\n" + "\n".join("• " + item["text"] for item in saved) + "\n\nИспользую эти факты в следующих чатах, пока память включена.", "memory_saved", model_name)
        return
    # Keep exact arithmetic, explicit safety replies and simple context handling deterministic.
    legacy, aside, route = prepare_reply(messages, personality)
    if route in {"calculator", "safety", "context"} or mode == "reference":
        yield {"type": "status", "text": aside}
        yield from text_events(legacy, route, model_name)
        return
    model = language_model(mode)
    context, sources = memory_context(memory, model.tokenizer, messages)
    if sources:
        yield source_event(sources)
        yield {"type": "status", "text": f"Добавляю в контекст {len(sources)} записей из памяти."}
    yield {"type": "status", "text": "Transformer учитывает порядок токенов и связи между ними в доступном контексте."}
    yield {"type": "status", "text": "Последовательно вычисляю вероятности следующих токенов ответа."}
    metadata, has_text = {}, False
    for event in model.generate_events(messages, context):
        if event["type"] == "token":
            has_text = has_text or bool(event["text"].strip())
            yield event
        else:
            metadata = event
    if not has_text:
        yield {"type": "token", "text": "Не удалось сформировать ответ. Попробуй уточнить вопрос или переключиться на справочный режим Mini."}
    if metadata.get("finish_reason") != "end":
        yield {"type": "status", "text": "Достигнут лимит генерации. Ответ может быть неполным."}
    yield {"type": "done", "model": model_name, "route": "transformer", "finish_reason": metadata.get("finish_reason", "end"),
           "tokens": metadata.get("tokens", 0), "context_tokens": metadata.get("context_tokens", 0)}
