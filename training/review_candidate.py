"""Raw generation comparison, separate from application tools and canned replies."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural.transformer import Transformer
from training.prepare_language import document_key

QUESTIONS = [
    ("user_regression", "Привет как дела?"),
    ("user_regression", "Привет, как дела?"),
    ("user_regression", "2+2 Сколько"),
    ("user_regression", "Пока"),
    ("fresh_paraphrase", "Привет! Как дела у тебя?"),
    ("fresh_paraphrase", "Увидимся завтра, пока!"),
    ("fresh_paraphrase", "Благодарю тебя за помощь."),
    ("fresh_question", "Зачем растениям солнечный свет?"),
    ("fresh_question", "Почему лёд тает в тёплой комнате?"),
    ("fresh_question", "Объясни разницу между списком и словарём в Python."),
    ("fresh_question", "Как сварить обычный рис в кастрюле?"),
    ("fresh_question", "В коробке три карандаша. Я добавил ещё два. Сколько их стало?"),
    ("fresh_question", "Ира пьёт чай, а Вова пьёт сок. Что пьёт Вова?"),
    ("fresh_question", "Скажи одним предложением, что такое дождь."),
    ("fresh_question", "Какая столица у Франции?"),
    ("fresh_question", "Как зовут моего соседа? Я не называл его имя."),
]


def review(directory, training_inputs, temperature):
    start = time.monotonic()
    model = Transformer(directory)
    load_time = time.monotonic() - start
    cases = []
    for kind, question in QUESTIONS:
        start = time.monotonic()
        answer = model.generate([{"role": "user", "content": question}],
                                temperature=temperature, top_p=.9, seed=43)
        row = dict(kind=kind, question=question, seen_exact_input_in_sft=document_key(question) in training_inputs,
                   seconds=round(time.monotonic() - start, 3), **answer)
        cases.append(row)
        print(json.dumps({"model": model.config["version"], "question": question, "answer": answer["text"]}, ensure_ascii=True), flush=True)
    return {"directory": str(directory), "config": model.config, "temperature": temperature,
            "load_seconds": load_time, "cases": cases}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=0.)
    args = p.parse_args()
    inputs = set()
    with (ROOT / "training-data/dialogue-budget/train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if len(row["messages"]) == 1:
                inputs.add(document_key(row["messages"][0]["content"]))
    results = [review(ROOT / path, inputs, args.temperature)
               for path in ("neural/checkpoint3", args.candidate)]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"raw_generation_only": True, "automatic_quality_claim": False,
                                 "note": "User regressions may be in authored training examples; exact matches are labelled. Fresh cases require human review. No calculator or memory handlers.",
                                 "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
