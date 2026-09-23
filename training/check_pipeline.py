"""CPU integration checks for the optional PyTorch training environment."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".training-libs"))
import numpy as np
import torch
from neural.tokenizer import Tokenizer, BOS, END
from training.scratch import DialogueData


class PipelineTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_sft_uses_own_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            tokenizer = Tokenizer.fit_bpe(["Привет, как дела? Всё хорошо. Пока! До встречи."] * 10, 120)
            tokenizer.save(data / "tokenizer.json")
            ids = ([BOS] + tokenizer.encode("Привет, как дела? Всё хорошо.") + [END]) * 30
            for split in ("train", "validation"):
                np.array(ids, dtype="<u4").tofile(data / (split + ".bin"))
            base = [sys.executable, str(ROOT / "training/scratch.py"), "--stage", "pretrain",
                    "--data", str(data), "--device", "cpu", "--width", "16", "--heads", "2",
                    "--layers", "1", "--context", "64", "--steps", "2", "--batch", "2",
                    "--accumulate", "1", "--eval-every", "1", "--eval-batches", "1"]

            def run(arguments, success=True):
                result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

            full, interrupted = root / "full", root / "interrupted"
            run(base + ["--output", str(full)])
            run(base + ["--output", str(interrupted), "--max-seconds", "0.000001"])
            run(base + ["--output", str(interrupted), "--resume"])
            left = torch.load(full / "last.pt", weights_only=True)
            right = torch.load(interrupted / "last.pt", weights_only=True)
            self.assertEqual(left["step"], 2)
            self.assertEqual(left["tokens"], right["tokens"])
            for name in left["model"]:
                torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
            row = dict(category="dialogue", messages=[{"role": "user", "content": "Привет, как дела?"}],
                       answer="Всё хорошо.", memory="")
            for split in ("train", "validation"):
                (data / (split + ".jsonl")).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            run([sys.executable, str(ROOT / "training/scratch.py"), "--stage", "sft", "--data", str(data),
                 "--initialize", str(full), "--output", str(root / "sft"), "--device", "cpu",
                 "--steps", "1", "--batch", "1", "--accumulate", "1", "--eval-batches", "1"])
            report = json.loads((root / "sft/progress.json").read_text())
            self.assertIn("initial_weights", report["data"])
            self.assertFalse(report["ready_for_chat"])
            with (data / "train.bin").open("ab") as stream:
                stream.write(np.array([END], dtype="<u4").tobytes())
            run(base + ["--output", str(interrupted), "--resume"], success=False)

    def test_sft_masks_question_and_retains_complete_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = Tokenizer.fit(["Вопрос? Ответ."])
            row = dict(category="dialogue", messages=[{"role": "user", "content": "Вопрос?"}], answer="Ответ.")
            for split in ("train", "validation"):
                (root / (split + ".jsonl")).write_text(json.dumps(row), encoding="utf-8")
            data = DialogueData(root, tokenizer, 64)
            _, labels = data.batch("train", 1, np.random.default_rng(1))
            actual = labels[0][labels[0] != -100].tolist()
            self.assertEqual(actual, tokenizer.encode("Ответ.") + [END])

    def test_sft_drops_overlong_question_instead_of_teaching_without_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = Tokenizer.fit(["Вопрос? Ответ."])
            short = dict(category="dialogue", messages=[{"role": "user", "content": "Вопрос?"}], answer="Ответ.")
            long = dict(short, messages=[{"role": "user", "content": "Вопрос? " * 100}])
            for split in ("train", "validation"):
                (root / (split + ".jsonl")).write_text(json.dumps(short) + "\n" + json.dumps(long), encoding="utf-8")
            data = DialogueData(root, tokenizer, 64)
            self.assertEqual(len(data.rows["train"]), 1)
            self.assertEqual(data.rejected["train"], 1)


if __name__ == "__main__":
    unittest.main()
