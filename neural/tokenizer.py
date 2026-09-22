"""A small reversible word/character tokenizer, fitted only on our corpus."""
import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, USER, ASSISTANT, END, MEMORY, UNKNOWN = range(7)
SPECIAL = ["<pad>", "<bos>", "<user>", "<assistant>", "<end>", "<memory>", "<unknown>"]


def pieces(text):
    return re.findall(r" ?[\w]+| ?[^\w\s]|\n| +|\t|\r", text, re.UNICODE)


class Tokenizer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.ids = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def fit(cls, texts, vocabulary=1800):
        texts = list(texts)
        chars = sorted(set("".join(texts)) | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;-_()\n"))
        vocab = SPECIAL + chars
        counts = Counter(piece for text in texts for piece in pieces(text))
        for piece, _ in counts.most_common():
            if piece not in vocab:
                vocab.append(piece)
            if len(vocab) >= vocabulary:
                break
        return cls(vocab)

    def encode(self, text):
        result = []
        for piece in pieces(text):
            if piece in self.ids:
                result.append(self.ids[piece])
            else:
                result.extend(self.ids.get(char, UNKNOWN) for char in piece)
        return result

    def decode(self, ids):
        return "".join(self.tokens[i] if i >= len(SPECIAL) else "�" if i == UNKNOWN else "" for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def prompt_tokens(tokenizer, messages, memory="", limit=224):
    """Prioritize the latest question, then recent turns, then retrieved memory.

    Role delimiters are numeric IDs, never parsed from user-controlled text.
    """
    latest = tokenizer.encode(messages[-1]["content"])
    # Keep the beginning and end of overlong questions, rather than just the end.
    if len(latest) > limit - 4:
        half = (limit - 4) // 2
        latest = latest[:half] + latest[-half:]
    tail = [USER] + latest + [END, ASSISTANT]
    remaining = max(0, limit - len(tail) - 1)
    memory_ids = tokenizer.encode(memory)[:min(80, remaining // 2)] if memory else []
    prefix = [MEMORY] + memory_ids + [END] if len(memory_ids) >= 1 and remaining >= 3 else []
    remaining -= len(prefix)
    turns = []
    for message in reversed(messages[:-1]):
        encoded = [USER if message["role"] == "user" else ASSISTANT] + tokenizer.encode(message["content"]) + [END]
        if len(encoded) > remaining:
            break
        turns.insert(0, encoded)
        remaining -= len(encoded)
    return [BOS] + prefix + [token for turn in turns for token in turn] + tail
