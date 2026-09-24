"""Compose independent existing questions within each split; no generated model text."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.tokenizer import Tokenizer, prompt_tokens, END


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model = Path(args.model)
    tokenizer = Tokenizer.load(model / "tokenizer.json")
    context = json.loads((model / "config.json").read_text())["context"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"method": "same-split independent-question composition, no external model", "splits": {}}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (Path(args.data) / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
        eligible = [r for r in rows if r.get("category") == "dialogue" and len(r["messages"]) == 1 and not r.get("memory")
                    and len(tokenizer.encode(r["messages"][0]["content"])) + len(tokenizer.encode(r["answer"])) < 200]
        rng = random.Random(917)
        additions = []
        for row in eligible:
            previous = rng.choice(eligible)
            if previous is row or previous["answer"] == row["answer"]:
                continue
            messages = [*previous["messages"], {"role": "assistant", "content": previous["answer"]}, *row["messages"]]
            if len(prompt_tokens(tokenizer, messages, limit=context+1)) + len(tokenizer.encode(row["answer"])) + 1 > context+1:
                continue
            additions.append({**row, "messages": messages, "source": "composed-topic-switch", "composition_sources": [previous.get("source"), row.get("source")]})
        (output / (split + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows + additions), encoding="utf-8")
        report["splits"][split] = {"original": len(rows), "added": len(additions)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
