"""Download pinned public text and prepare disjoint, disk-backed language data.

No pretrained weights/tokenizers, no private chat history. Wikipedia attribution
is retained per document. Outputs are research data, not deployable chat weights.
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".training-libs"))
import numpy as np
from neural.tokenizer import Tokenizer, BOS, END, UNKNOWN

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
REPO = "wikimedia/wikipedia"


def document_key(text):
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def document_split(key):
    bucket = int(key[:8], 16) % 100
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(shards, output, vocabulary=8192, sample_chars=2_000_000):
    import pyarrow.parquet as pq
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of the data for an existing training run.
    if (output / "manifest.json").exists():
        raise ValueError("Dataset exists; choose a new --output directory")
    counts, seen, sample = Counter(), set(), []
    sampled = 0
    with ExitStack() as stack:
        streams = {s: stack.enter_context((output / (s + ".jsonl")).open("w", encoding="utf-8"))
                   for s in ("train", "validation", "test")}
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    text = row["text"].strip()
                    if len(text) < 200 or not re.search(r"[А-Яа-яЁё]", text):
                        counts["filtered"] += 1
                        continue
                    key = document_key(text)
                    if key in seen:
                        counts["duplicate"] += 1
                        continue
                    seen.add(key)
                    split = document_split(key)
                    record = {"text": text, "id": row["id"], "url": row["url"],
                              "title": row["title"], "sha256": key}
                    streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    if split == "train" and sampled < sample_chars:
                        sample.append(text[:sample_chars - sampled])
                        sampled += len(sample[-1])
    print(f"Documents: {dict(counts)}. Fitting own BPE on {sampled} train characters.", flush=True)
    tokenizer = Tokenizer.fit_bpe(sample, vocabulary=vocabulary)
    tokenizer.save(output / "tokenizer.json")
    manifest = {"source": REPO, "revision": REVISION, "language": "ru", "snapshot": "20231101",
                "license": "CC-BY-SA-3.0 / GFDL; attribution in split JSONL url/title/id",
                "source_shards": {Path(p).name: sha256(p) for p in shards},
                "documents": dict(counts), "tokenizer_train_characters": sampled,
                "vocabulary": len(tokenizer.tokens), "dtype": "uint32", "splits": {}}
    for split in ("train", "validation", "test"):
        tokens = unknown = 0
        path = output / (split + ".bin")
        with (output / (split + ".jsonl")).open(encoding="utf-8") as source, path.open("wb") as destination:
            for line in source:
                ids = [BOS] + tokenizer.encode(json.loads(line)["text"]) + [END]
                np.asarray(ids, dtype="<u4").tofile(destination)
                tokens += len(ids)
                unknown += ids.count(UNKNOWN)
        manifest["splits"][split] = {"tokens": tokens, "unknown_tokens": unknown, "sha256": sha256(path)}
        print(f"{split}: {tokens:,} tokens ({unknown:,} unknown)", flush=True)
    manifest["tokenizer_sha256"] = sha256(output / "tokenizer.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", default="training-data/language")
    parser.add_argument("--vocabulary", type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.shards <= 21:
        parser.error("--shards must be between 1 and 21")
    cache = ROOT / "training-data/wikipedia"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(args.shards):
        filename = f"train-{index:05d}-of-00021.parquet"
        path = cache / filename
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/20231101.ru/{filename}"
            print(f"Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Kraken-scratch-training/4"})
            temp = path.with_suffix(".partial")
            with urllib.request.urlopen(request, timeout=180) as source, temp.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
            temp.replace(path)
        paths.append(path)
    prepare(paths, ROOT / args.output, args.vocabulary)


if __name__ == "__main__":
    main()
