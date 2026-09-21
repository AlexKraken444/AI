"""Shared feature extraction and pure-Python inference. No pretrained models."""
import hashlib
import json
import math
import re
from pathlib import Path

FEATURES = 384
HIDDEN = 48


def words(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))


def features(text):
    vector = [0.0] * FEATURES
    for word in words(text):
        padded = "^" + word + "$"
        tokens = ["w:" + word] + [padded[i:i + 3] for i in range(len(padded) - 2)]
        for token in tokens:
            digest = hashlib.blake2s(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % FEATURES] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Network:
    def __init__(self, path=None):
        path = path or Path(__file__).with_name("weights.json")
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def predict(self, text):
        x = features(text)
        d = self.data
        h = [math.tanh(b + sum(v * w for v, w in zip(x, row)))
             for row, b in zip(d["w1"], d["b1"])]
        logits = [b + sum(v * w for v, w in zip(h, row))
                  for row, b in zip(d["w2"], d["b2"])]
        exps = [math.exp(v - max(logits)) for v in logits]
        probabilities = [v / sum(exps) for v in exps]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return d["labels"][best], probabilities[best]
