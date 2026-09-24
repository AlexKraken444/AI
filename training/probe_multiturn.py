"""Record raw replies to changing topics; no classifier or prepared answers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.transformer import Transformer

SCENARIOS = [
    ["Привет, как дела?", "Что такое Python?", "Теперь другая тема: столица Франции?"],
    ["Кошка сидит под столом. Где кошка?", "Теперь кошка вышла во двор. Где она теперь?"],
    ["Меня зовут Алекс.", "Как меня зовут?", "Сколько будет 2+2?"],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = Transformer(args.model)
    scenarios = []
    for questions in SCENARIOS:
        history, results = [], []
        for question in questions:
            history.append({"role": "user", "content": question})
            result = model.generate(history, max_tokens=96, temperature=0, seed=43)
            results.append({"question": question, **result})
            history.append({"role": "assistant", "content": result["text"]})
            print(json.dumps(results[-1], ensure_ascii=True), flush=True)
        scenarios.append(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": model.config, "requires_human_review": True,
                                 "scenarios": scenarios}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
