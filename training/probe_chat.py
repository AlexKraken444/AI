"""Record raw model answers for human review; no templates or calculator routes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural.transformer import Transformer

QUESTIONS = [
    "Привет как дела?", "Привет, как дела?", "2+2 Сколько", "Пока",
    "Почему зимой холоднее, чем летом?",
    "Объясни, чем список отличается от словаря в Python.",
    "Маша дала Пете две книги. У Пети была одна книга. Сколько книг стало у Пети?",
    "Кошка сидит под столом. Где кошка?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.)
    args = parser.parse_args()
    model = Transformer(args.model)
    cases = []
    for question in QUESTIONS:
        answer = model.generate([{"role": "user", "content": question}],
                                temperature=args.temperature, top_p=.9, seed=43)
        cases.append({"question": question, **answer})
        print(json.dumps(cases[-1], ensure_ascii=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": model.config, "temperature": args.temperature,
                                 "requires_human_review": True, "cases": cases},
                                ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
