"""Explicit, dated reference answers. This is retrieval, not neural generation."""
import re
from difflib import SequenceMatcher

SOURCE = "https://kremlin.ru/structure/president"
VERIFIED = "2026-09-24"
VOCABULARY = ("президент", "россии", "российской", "федерации", "зовут", "сейчас", "кто", "какой", "столица")


def normalize_query(text):
    words = re.findall(r"[а-яёa-z]+|\d+", text.lower().replace("ё", "е"))
    result = []
    for word in words:
        word = {"призедент": "президент", "президентa": "президент"}.get(word, word)
        candidates = [(SequenceMatcher(None, word, target).ratio(), target) for target in VOCABULARY
                      if len(word) >= 5 and abs(len(word) - len(target)) <= 2]
        candidates.sort(reverse=True)
        if candidates and candidates[0][0] >= .78 and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= .1):
            word = candidates[0][1]
        result.append(word)
    return result


def fact_reply(text):
    words = normalize_query(text)
    tokens = set(words)
    # A name lookup only: dates, comparisons and policy questions must not be
    # answered with the current office holder merely because keywords match.
    allowed = set(VOCABULARY) | {"рф", "как", "имя", "фамилия", "отчество", "нынешний", "действующий", "у", "нас", "в", "на", "данный", "момент", "пожалуйста", "скажи"}
    russia = "россии" in tokens or "рф" in tokens or {"российской", "федерации"} <= tokens
    if "президент" in tokens and russia and tokens <= allowed:
        return ("Президент России — Владимир Владимирович Путин.\n\n"
                f"Источник: {SOURCE}\nПроверено: {VERIFIED}. Это справочная запись, а не проверка новостей в реальном времени.")
    if tokens & {"президент", "столица"}:
        return "Для этого вопроса у меня нет подходящего проверенного факта. Не буду придумывать имя, дату или событие. Уточни вопрос или предоставь источник."
    return None
