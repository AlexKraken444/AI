"""Causal multi-head Transformer inference in NumPy, with per-request KV cache.

All weights are locally trained from random initialization. No language model API.
PyTorch is needed only by train_transformer.py, never by the Vercel function.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
from neural.tokenizer import Tokenizer, PAD, BOS, USER, ASSISTANT, END, MEMORY, UNKNOWN, prompt_tokens


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def select_token(scores, rng, temperature=0., top_p=.9):
    """Greedy or nucleus sampling; returns token and untempered probability.

    Probability describes the model's prediction, not truth or answer quality.
    """
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid temperature/top_p")
    raw = softmax(scores)
    if temperature == 0:
        token = int(raw.argmax())
    else:
        distribution = softmax(scores.astype(np.float64) / temperature)
        order = np.argsort(-distribution, kind="stable")
        cumulative = np.cumsum(distribution[order])
        size = min(len(order), int(np.searchsorted(cumulative, top_p)) + 1)
        candidates = order[:size]
        weights = distribution[candidates]
        token = int(rng.choice(candidates, p=weights / weights.sum()))
    return token, float(raw[token])


class Transformer:
    def __init__(self, directory=None):
        root = Path(directory) if directory else Path(__file__).parent / "checkpoint"
        self.config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.load(root / "tokenizer.json")
        with np.load(root / "model.npz", allow_pickle=False) as archive:
            self.weights = {key: archive[key].astype(np.float32) for key in archive.files}

    def norm(self, x, name):
        variance = x.var(axis=-1, keepdims=True)
        return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(variance + 1e-5) * self.weights[name + ".weight"] + self.weights[name + ".bias"]

    def linear(self, x, name):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def forward(self, token_ids, cache=None):
        """Return next-token logits, new KV cache and final token embeddings."""
        d, heads = self.config["width"], self.config["heads"]
        offset = cache[0][0].shape[1] if cache else 0
        length = len(token_ids)
        if not length or offset + length > self.config["context"]:
            raise ValueError("Context length exceeded")
        x = self.weights["token.weight"][token_ids] + self.weights["position.weight"][offset:offset + length]
        next_cache = []
        for layer in range(self.config["layers"]):
            prefix = f"blocks.{layer}"
            h = self.norm(x, prefix + ".norm1")
            q, k, v = np.split(self.linear(h, prefix + ".qkv"), 3, axis=-1)
            q, k, v = [a.reshape(length, heads, d // heads).transpose(1, 0, 2) for a in (q, k, v)]
            if cache:
                k = np.concatenate([cache[layer][0], k], axis=1)
                v = np.concatenate([cache[layer][1], v], axis=1)
            scores = q @ k.transpose(0, 2, 1) / math.sqrt(d // heads)
            mask = np.arange(k.shape[1])[None, :] > (offset + np.arange(length))[:, None]
            attention = softmax(np.where(mask[None, :, :], -1e9, scores))
            out = (attention @ v).transpose(1, 0, 2).reshape(length, d)
            x = x + self.linear(out, prefix + ".projection")
            h = self.linear(self.norm(x, prefix + ".norm2"), prefix + ".ff1")
            h = .5 * h * (1 + np.tanh(math.sqrt(2 / math.pi) * (h + .044715 * h ** 3)))
            x = x + self.linear(h, prefix + ".ff2")
            next_cache.append((k, v))
        hidden = self.norm(x, "final_norm")
        logits = hidden @ self.weights["head.weight"].T
        return logits, next_cache, hidden

    def generate_events(self, messages, memory="", max_tokens=128, time_limit=12,
                        temperature=0., top_p=.9, seed=None):
        if not 1 <= max_tokens <= self.config["context"] - 16:
            raise ValueError("Invalid output token budget")
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid temperature/top_p")
        rng = np.random.default_rng(seed)
        ids = prompt_tokens(self.tokenizer, messages, memory, limit=self.config["context"] - max_tokens)
        logits, cache, _ = self.forward(ids)
        output, probabilities = [], []
        deadline = time.monotonic() + time_limit
        finish_reason = "length"
        for _ in range(max_tokens):
            if time.monotonic() > deadline:
                finish_reason = "timeout"
                break
            scores = logits[-1].copy()
            scores[[PAD, BOS, USER, ASSISTANT, MEMORY, UNKNOWN]] = -1e9
            # Prevent identical trigram loops without increasing randomness.
            if len(output) >= 2:
                for i in range(len(output) - 2):
                    if output[i:i + 2] == output[-2:]:
                        scores[output[i + 2]] = -1e9
            token, probability = select_token(scores, rng, temperature, top_p)
            if token == END:
                finish_reason = "end"
                break
            output.append(token)
            probabilities.append(probability)
            yield {"type": "token", "text": self.tokenizer.decode([token])}
            logits, cache, _ = self.forward([token], cache)
        yield {"type": "generation_done", "tokens": len(output), "finish_reason": finish_reason,
               "mean_probability": sum(probabilities) / max(1, len(probabilities)), "context_tokens": len(ids)}

    def generate(self, messages, memory="", max_tokens=128, time_limit=12,
                 temperature=0., top_p=.9, seed=None):
        text, metadata = "", {}
        for event in self.generate_events(messages, memory, max_tokens, time_limit, temperature, top_p, seed):
            if event["type"] == "token":
                text += event["text"]
            else:
                metadata = {key: value for key, value in event.items() if key != "type"}
        return {"text": text.strip(), **metadata}
