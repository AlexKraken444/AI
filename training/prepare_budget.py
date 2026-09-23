"""Reuse a bounded Wikipedia token prefix and prepare attributed Russian SFT data."""
import io
import json
import shutil
from collections import Counter
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".training-libs"))
import numpy as np
import zstandard
from neural.tokenizer import Tokenizer, BOS, END
from training.prepare_language import sha256, document_key, document_split

REVISION = "460b1f3312aa21ef774e916e532a9576f7938a0d"
TRAIN_TOKENS = 71_135_516


def language():
    source = ROOT / "training-data/language"
    dest = ROOT / "training-data/language-budget"
    dest.mkdir(exist_ok=True)
    if (dest / "manifest.json").exists():
        return
    shutil.copyfile(source / "tokenizer.json", dest / "tokenizer.json")
    # The previous interrupted encoder produced a valid prefix of uint32 tokens.
    # It is deliberately a new, bounded dataset, never labelled the full corpus.
    size = (source / "train.bin").stat().st_size
    if size % 4 or size < 4 * TRAIN_TOKENS:
        raise ValueError("Invalid prepared prefix")
    with (source / "train.bin").open("rb") as src, (dest / "train.bin").open("wb") as dst:
        remaining = TRAIN_TOKENS * 4
        while remaining:
            chunk = src.read(min(1024 * 1024, remaining))
            dst.write(chunk)
            remaining -= len(chunk)
    tokenizer = Tokenizer.load(dest / "tokenizer.json")
    for split in ("validation", "test"):
        with (source / (split + ".jsonl")).open(encoding="utf-8") as stream, (dest / (split + ".bin")).open("wb") as out:
            for line in stream:
                ids = [BOS] + tokenizer.encode(json.loads(line)["text"]) + [END]
                np.asarray(ids, dtype="<u4").tofile(out)
    report = {"source": "wikimedia/wikipedia", "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
              "license": "CC-BY-SA-3.0 / GFDL", "scope": "bounded train token prefix; full disjoint validation/test documents",
              "attribution": "training-data/language/{train,validation,test}.jsonl retain article URLs/titles/IDs",
              "splits": {s: {"tokens": (dest / (s + ".bin")).stat().st_size // 4,
                              "sha256": sha256(dest / (s + ".bin"))} for s in ("train", "validation", "test")}}
    (dest / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def authored_dialogues():
    groups = [
        (["Привет", "Здравствуй", "Здравствуйте", "Добрый день", "Приветик", "Добрый вечер"],
         ["Привет! Чем могу помочь?", "Привет! Что хочешь обсудить?"]),
        (["Как дела", "Как ты", "Как у тебя дела", "Как настроение", "Как поживаешь"],
         ["Готов помочь. А как у тебя дела?", "Всё хорошо, спасибо! Что у тебя нового?"]),
        (["Пока", "До свидания", "До встречи", "Увидимся", "Мне пора", "Спокойной ночи"],
         ["До встречи!", "Пока! Буду рад продолжить разговор."]),
        (["Спасибо", "Спасибо за помощь", "Благодарю", "Большое спасибо"],
         ["Пожалуйста!", "Рад помочь!"]),
    ]
    greetings = groups[0][0]
    combined = [g + separator + q.lower() for g in greetings for q in groups[1][0] for separator in (", ", "! ", " ")]
    groups.append((combined, ["Привет! Всё хорошо, готов помочь. Как у тебя дела?"]))
    for index, (questions, answers) in enumerate(groups):
        for i, question in enumerate(questions):
            for punctuation in ("", "!", "?"):
                yield {"messages": [{"role": "user", "content": question + punctuation}],
                       "answer": answers[i % len(answers)], "category": "dialogue", "memory": "",
                       "source": "authored-smalltalk", "group": f"smalltalk:{index}:{question.casefold()}"}


def instructions():
    output = ROOT / "training-data/dialogue-budget"
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        return
    splits = {s: [] for s in ("train", "validation", "test")}
    rejected, seen = Counter(), set()
    archive = ROOT / "training-data/ru_turbo_alpaca.jsonl.zst"
    if not archive.exists():
        url = f"https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca/resolve/{REVISION}/{archive.name}"
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=90) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(archive)
    if sha256(archive) != "5b7ceb2bd34a025d4618222d273f36a129910dfcb148e75d366cb992a74eb9e7":
        raise ValueError("Instruction corpus checksum mismatch")
    with archive.open("rb") as compressed, io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(compressed), encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row.get("label") in ("bad_task", "bad_output"):
                rejected["negative_review"] += 1
                continue
            question = row["instruction"].strip()
            extra = (row.get("input") or "").strip()
            if extra and extra not in ("<noinput>", "<no input>"):
                question += "\n\n" + extra
            answer = row["output"].strip()
            if not question or not answer or len(question) > 6000 or len(answer) > 6000:
                rejected["empty_or_long"] += 1
                continue
            if re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", question + answer, re.I):
                rejected["email"] += 1
                continue
            key = document_key(question)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            # 90/5/5 split by normalized input, before any tokenizer length filtering.
            bucket = int(key[:8], 16) % 20
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            splits[split].append({"messages": [{"role": "user", "content": question}], "answer": answer,
                                  "memory": "", "category": "dialogue", "source": "IlyaGusev/ru_turbo_alpaca",
                                  "source_id": index, "review_label": row.get("label"), "group": key})
    for split in splits:
        with (ROOT / "training-data" / (split + ".jsonl")).open(encoding="utf-8") as stream:
            splits[split].extend(json.loads(line) for line in stream if '"category": "dialogue"' in line)
    for row in authored_dialogues():
        bucket = int(document_key(row["group"])[:8], 16) % 20
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        splits[split].append(row)
    # Remove exact duplicate inputs across all three sources, including conflicts
    # between the original OASST split and the new synthetic corpus.
    all_inputs = set()
    for split, rows in splits.items():
        unique = []
        for row in rows:
            key = document_key(json.dumps([row["messages"], row.get("memory", "")], ensure_ascii=False))
            if key in all_inputs:
                rejected["cross_source_duplicate"] += 1
                continue
            all_inputs.add(key)
            unique.append(row)
        splits[split] = unique
    report = {"source": "IlyaGusev/ru_turbo_alpaca", "author": "Ilya Gusev and contributors",
              "revision": REVISION, "license": "CC-BY-4.0", "archive_sha256": sha256(archive),
              "note": "Synthetic instruction pairs; negative reviews excluded, unlabeled rows not independently verified. Also OASST1 Apache-2.0 and authored smalltalk.",
              "rejected": dict(rejected), "splits": {}}
    for split, rows in splits.items():
        path = output / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        report["splits"][split] = {"examples": len(rows), "by_source": dict(Counter(r["source"] for r in rows)), "sha256": sha256(path)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    language()
    instructions()
