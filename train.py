"""Train our MLP from random weights using explicit backpropagation and Adam.

NumPy performs matrix arithmetic only; no neural-network framework is used.
Run: python -m pip install -r requirements-train.txt && python train.py
"""
import json
from pathlib import Path
import numpy as np
from neural.network import FEATURES, HIDDEN, features

ROOT = Path(__file__).parent / "neural"


def train():
    intents = json.loads((ROOT / "intents.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    labels = [item["id"] for item in intents]
    examples = [(text, i) for i, item in enumerate(intents) for text in item["patterns"]]
    x = np.asarray([features(t) for t, _ in examples])
    y = np.eye(len(labels))[[i for _, i in examples]]
    params = [rng.normal(0, .12, (HIDDEN, FEATURES)), np.zeros(HIDDEN),
              rng.normal(0, .12, (len(labels), HIDDEN)), np.zeros(len(labels))]
    moment = [np.zeros_like(p) for p in params]
    velocity = [np.zeros_like(p) for p in params]
    for step in range(1, 501):
        w1, b1, w2, b2 = params
        hidden = np.tanh(x @ w1.T + b1)
        logits = hidden @ w2.T + b2
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = exp / exp.sum(axis=1, keepdims=True)
        delta = (prob - y) / len(x)
        dh = (delta @ w2) * (1 - hidden ** 2)
        grads = [dh.T @ x + .0001 * w1, dh.sum(axis=0),
                 delta.T @ hidden + .0001 * w2, delta.sum(axis=0)]
        for j, grad in enumerate(grads):
            moment[j] = .9 * moment[j] + .1 * grad
            velocity[j] = .999 * velocity[j] + .001 * grad ** 2
            params[j] -= .012 * (moment[j] / (1 - .9 ** step)) / (np.sqrt(velocity[j] / (1 - .999 ** step)) + 1e-8)
    accuracy = float((prob.argmax(axis=1) == y.argmax(axis=1)).mean())
    result = dict(zip(("w1", "b1", "w2", "b2"), [p.round(7).tolist() for p in params]))
    result.update(labels=labels, version="kraken-mini-1", seed=42,
                  examples=len(examples), training_accuracy=accuracy,
                  parameters=sum(p.size for p in params))
    (ROOT / "weights.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Trained {result['parameters']} parameters on {len(examples)} examples; training accuracy {accuracy:.1%} (not a generalization benchmark).")


if __name__ == "__main__":
    train()
