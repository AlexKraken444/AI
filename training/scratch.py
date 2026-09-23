"""Two-stage training of our own Transformer: language pretraining then SFT.

CUDA/CPU, gradient accumulation, mixed precision, exact resume state, held-out
token loss, and portable NumPy export. Never overwrites deployed checkpoints.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".training-libs"))
import numpy as np
import torch
from torch.nn import functional as F
from train_transformer import LanguageModel
from neural.tokenizer import Tokenizer, END, prompt_tokens
from training.prepare_language import sha256


class LanguageData:
    def __init__(self, directory, context):
        self.context = context
        self.streams = {s: np.memmap(Path(directory) / (s + ".bin"), dtype="<u4", mode="r")
                        for s in ("train", "validation")}
        if any(len(data) <= context for data in self.streams.values()):
            raise ValueError("Both splits must exceed the context length")

    def batch(self, split, size, rng):
        data = self.streams[split]
        starts = rng.integers(0, len(data) - self.context, size=size)
        rows = np.stack([data[i:i + self.context + 1] for i in starts]).astype(np.int64)
        return torch.from_numpy(rows[:, :-1].copy()), torch.from_numpy(rows[:, 1:].copy())


class DialogueData:
    def __init__(self, directory, tokenizer, context, basic_fraction=0.):
        self.rows = {}
        self.rejected = {}
        self.basic_indices = []
        self.basic_fraction = basic_fraction
        for split in ("train", "validation"):
            rows, rejected = [], 0
            with (Path(directory) / (split + ".jsonl")).open(encoding="utf-8") as stream:
                for line in stream:
                    e = json.loads(line)
                    # Real dialogues only; do not swamp them with synthetic fact tasks.
                    if e.get("category") != "dialogue":
                        continue
                    answer = tokenizer.encode(e["answer"]) + [END]
                    if len(answer) > context - 8:
                        rejected += 1
                        continue
                    prompt = prompt_tokens(tokenizer, e["messages"], e.get("memory", ""),
                                           limit=context + 1)
                    if len(prompt) + len(answer) > context + 1:
                        rejected += 1
                        continue
                    if split == "train" and e.get("source") == "authored-smalltalk":
                        self.basic_indices.append(len(rows))
                    rows.append((prompt + answer, [-100] * (len(prompt) - 1) + answer))
            if not rows:
                raise ValueError(f"No complete dialogues fit {split}; increase context or supply more data")
            self.rows[split], self.rejected[split] = rows, rejected
        print(f"SFT counts: { {k: len(v) for k, v in self.rows.items()} }; too long: {self.rejected}", flush=True)

    def batch(self, split, size, rng):
        rows = self.rows[split]
        indices = rng.integers(0, len(rows), size=size)
        if split == "train" and self.basic_indices and self.basic_fraction:
            mask = rng.random(size) < self.basic_fraction
            indices[mask] = rng.choice(self.basic_indices, size=int(mask.sum()))
        chosen = [rows[i] for i in indices]
        length = max(len(y) for _, y in chosen)
        x = torch.zeros((size, length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (ids, labels) in enumerate(chosen):
            x[i, :len(labels)] = torch.tensor(ids[:-1])
            y[i, :len(labels)] = torch.tensor(labels)
        return x, y


def validation(model, data, device, size, batches, amp):
    # Fixed independent RNG means comparable validation windows and no training RNG drift.
    rng = np.random.default_rng(191)
    total, count = 0., 0
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, y = data.batch("validation", size, rng)
            x, y = x.to(device), y.to(device)
            with amp():
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
            total += loss.item()
            count += int((y != -100).sum())
    return total / count


def export(model, output):
    temp = output / "candidate.npz"
    np.savez_compressed(temp, **{k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()})
    temp.replace(output / "model.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initialize", help="Our pretrained export directory, required for SFT")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=0, help="Stop after saving a resumable checkpoint")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--basic-fraction", type=float, default=0., help="SFT-only extra sampling probability for authored conversational basics")
    args = p.parse_args()
    if min(args.steps, args.batch, args.accumulate, args.eval_every, args.eval_batches,
           args.width, args.layers, args.heads, args.context, args.threads) < 1:
        p.error("Counts must be positive")
    if args.width % args.heads or args.context < 16 or args.lr <= 0 or args.max_seconds < 0 or not 0 <= args.basic_fraction <= .5:
        p.error("Invalid architecture or optimizer settings")
    if (args.stage == "sft") != bool(args.initialize):
        p.error("SFT requires --initialize; pretraining always starts from random weights")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; install a CUDA build of PyTorch")
    torch.set_num_threads(args.threads)
    torch.manual_seed(43)
    rng = np.random.default_rng(43)
    output, directory = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        p.error("Output must be empty for a new run")
    origin = Path(args.initialize) if args.initialize else directory
    tokenizer = Tokenizer.load(origin / "tokenizer.json")
    if args.initialize:
        config = json.loads((origin / "config.json").read_text(encoding="utf-8"))
        config = {**config, "version": "kraken-scratch-sft"}
    else:
        config = dict(version="kraken-scratch-pretrain", width=args.width, layers=args.layers,
                      heads=args.heads, context=args.context, vocabulary=len(tokenizer.tokens), seed=43)
    data = (LanguageData(directory, config["context"]) if args.stage == "pretrain" else
            DialogueData(directory, tokenizer, config["context"], args.basic_fraction))
    suffix = ".bin" if args.stage == "pretrain" else ".jsonl"
    signature = {s: sha256(directory / (s + suffix)) for s in ("train", "validation")}
    signature["tokenizer"] = sha256(origin / "tokenizer.json")
    if args.initialize:
        signature["initial_weights"] = sha256(origin / "model.npz")
    model = LanguageModel(config).to(device)
    config["parameters"] = sum(v.numel() for v in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device == "cuda" else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
    start_step, seen, best, history = 0, 0, float("inf"), []
    run_settings = {k: getattr(args, k) for k in ("stage", "steps", "batch", "accumulate", "lr", "eval_batches")}
    if args.stage == "sft":
        run_settings["basic_fraction"] = args.basic_fraction
    if args.resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != config or state["data"] != signature or state["settings"] != run_settings:
            p.error("Resume requires the same architecture, data, tokenizer and training schedule")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, seen, best, history = state["step"], state["tokens"], state["best"], state["history"]
    elif args.initialize:
        with np.load(origin / "model.npz", allow_pickle=False) as weights:
            model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files})
    tokenizer.save(output / "tokenizer.json")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"{device}: {config['parameters']:,} random-origin parameters; stage={args.stage}; resume={start_step}", flush=True)
    if not history:
        initial = validation(model, data, device, args.batch, args.eval_batches, amp)
        history.append({"step": 0, "validation_nll": initial})
        print(f"Initial held-out NLL: {initial:.4f}", flush=True)
    begin, session_tokens = time.monotonic(), seen
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        lr = args.lr * min(1., step / min(100, args.steps)) * (.1 + .9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(args.accumulate):
            x, y = data.batch("train", args.batch, rng)
            x, y = x.to(device), y.to(device)
            with amp():
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing to save corrupted weights")
            scaler.scale(loss / args.accumulate).backward()
            loss_sum += loss.item() / args.accumulate
            seen += int((y != -100).sum())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        elapsed = time.monotonic() - begin
        timed_out = args.max_seconds > 0 and elapsed >= args.max_seconds
        if step % 25 == 0:
            print(f"step={step} train_nll={loss_sum:.4f} tokens={seen:,} tokens/s={(seen-session_tokens)/elapsed:.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps or timed_out:
            score = validation(model, data, device, args.batch, args.eval_batches, amp)
            history.append({"step": step, "validation_nll": score, "train_nll": loss_sum, "tokens": seen})
            if score < best:
                best = score
                export(model, output)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                         config=config, data=signature, settings=run_settings, step=step, tokens=seen,
                         best=best, history=history)
            temp = output / "last.tmp"
            torch.save(state, temp)
            temp.replace(output / "last.pt")
            report = dict(stage=args.stage, device=device, parameters=config["parameters"], step=step,
                          planned_steps=args.steps, tokens=seen, best_validation_nll=best, history=history,
                          session_training_seconds=round(time.monotonic() - begin, 2),
                          data=signature, ready_for_chat=False,
                          note="NLL is next-token prediction loss, not conversational quality. Evaluate chat separately.")
            (output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"validation_nll={score:.4f}; best={best:.4f}; checkpoint saved", flush=True)
        if timed_out:
            break
    # Verify the best exported checkpoint, including the CUDA -> NumPy path.
    from neural.transformer import Transformer
    runtime = Transformer(output)
    with np.load(output / "model.npz", allow_pickle=False) as weights:
        model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files})
    model.eval()
    ids = [1] + tokenizer.encode("Привет, как дела?")
    with torch.no_grad():
        reference = model(torch.tensor([ids], device=device))[0].float().cpu().numpy()
    actual, _, _ = runtime.forward(ids)
    np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-3)
    print("Export parity passed. This checkpoint is not automatically published.", flush=True)


if __name__ == "__main__":
    main()
