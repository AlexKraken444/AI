"""Train a hand-built causal Transformer from scratch; export portable NumPy weights."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".training-libs").exists():
    sys.path.insert(0, str(ROOT / ".training-libs"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural.tokenizer import Tokenizer, END, prompt_tokens
from training.corpus import corpus


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, width * 3)
        self.ff2 = nn.Linear(width * 3, width)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [item.reshape(b, t, self.heads, d // self.heads).transpose(1, 2) for item in (q, k, v)]
        # Explicit causal self-attention; no pretrained architecture or weights.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(b, t, d)
        x = x + self.projection(attended)
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token = nn.Embedding(config["vocabulary"], config["width"])
        self.position = nn.Embedding(config["context"], config["width"])
        self.blocks = nn.ModuleList([Block(config["width"], config["heads"]) for _ in range(config["layers"])])
        self.final_norm = nn.LayerNorm(config["width"])
        self.head = nn.Linear(config["width"], config["vocabulary"], bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.025)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train, validation = corpus()
    tokenizer = Tokenizer.fit([text for e in train for text in [e["answer"], e["memory"]] + [m["content"] for m in e["messages"]]])
    config = {"version": "kraken-context-2", "width": 128, "heads": 4, "layers": 3,
              "context": 384, "vocabulary": len(tokenizer.tokens), "seed": 42}
    model = LanguageModel(config)
    if args.resume:
        with np.load(ROOT / "neural/checkpoint/model.npz", allow_pickle=False) as archive:
            model.load_state_dict({key: torch.from_numpy(archive[key]) for key in archive.files})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)

    def encode(example):
        prompt = prompt_tokens(tokenizer, example["messages"], example["memory"], limit=160)
        answer = tokenizer.encode(example["answer"])[:120] + [END]
        ids = prompt + answer
        labels = [-100] * (len(prompt) - 1) + answer
        return ids[:-1], labels

    training = [encode(e) for e in train]
    testing = [encode(e) for e in validation]
    dialogue_indices = [i for i, e in enumerate(train) if len(tokenizer.encode(e["answer"])) > 18]
    def batch(examples, indices):
        rows = [examples[i] for i in indices]
        length = max(len(row[0]) for row in rows)
        x = torch.zeros((len(rows), length), dtype=torch.long)
        y = torch.full((len(rows), length), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(rows):
            x[i, :len(ids)] = torch.tensor(ids)
            y[i, :len(labels)] = torch.tensor(labels)
        return x, y

    root = ROOT / "neural/checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    tokenizer.save(root / "tokenizer.json")
    config["parameters"] = sum(p.numel() for p in model.parameters())
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.monotonic()
    print(f"Training {config['parameters']:,} parameters; {len(train)} train, {len(validation)} held-out examples", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        indices = np.concatenate([rng.choice(dialogue_indices, size=args.batch // 3), rng.integers(len(training), size=args.batch - args.batch // 3)])
        x, y = batch(training, indices)
        lr = .0015 * min(1, step / 80) * (.2 + .8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        token_loss = F.cross_entropy(logits.reshape(-1, len(tokenizer.tokens)), y.reshape(-1), reduction="none").reshape_as(y)
        loss = (token_loss.sum(dim=1) / (y != -100).sum(dim=1).clamp_min(1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(testing, np.arange(min(64, len(testing))))
                val_loss = float(F.cross_entropy(model(vx).reshape(-1, len(tokenizer.tokens)), vy.reshape(-1)))
            print(f"step {step} train={float(loss.detach()):.3f} validation={val_loss:.3f} elapsed={time.monotonic()-start:.1f}s", flush=True)
            np.savez_compressed(root / "model.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    model.eval()
    total_loss, total_tokens = 0., 0
    with torch.no_grad():
        for offset in range(0, len(testing), 32):
            vx, vy = batch(testing, range(offset, min(offset + 32, len(testing))))
            total_loss += float(F.cross_entropy(model(vx).reshape(-1, len(tokenizer.tokens)), vy.reshape(-1), reduction="sum"))
            total_tokens += int((vy != -100).sum())
    metrics = {"steps": args.steps, "training_examples": len(train), "validation_examples": len(validation),
               "validation_token_loss": total_loss / total_tokens, "elapsed_seconds": round(time.monotonic() - start, 2),
               "parameters": config["parameters"], "note": "Small original/synthetic corpus; this is not a general reasoning benchmark."}
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Numerical parity check between training implementation and deployed NumPy implementation.
    from neural.transformer import Transformer
    runtime = Transformer(root)
    ids = prompt_tokens(tokenizer, [{"role": "user", "content": "Что такое Python?"}])
    with torch.no_grad():
        reference = model(torch.tensor([ids]))[0].numpy()
    actual, _, _ = runtime.forward(ids)
    np.testing.assert_allclose(actual, reference, atol=3e-4, rtol=3e-4)
    print("NumPy/PyTorch parity passed", flush=True)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
