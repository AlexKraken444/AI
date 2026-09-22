"""A small reversible word/character tokenizer, fitted only on our corpus."""
import json
import re
import heapq
from collections import Counter
from pathlib import Path

PAD, BOS, USER, ASSISTANT, END, MEMORY, UNKNOWN = range(7)
SPECIAL = ["<pad>", "<bos>", "<user>", "<assistant>", "<end>", "<memory>", "<unknown>"]


def pieces(text):
    return re.findall(r" ?[\w]+| ?[^\w\s]|\n| +|\t|\r", text, re.UNICODE)


class Tokenizer:
    def __init__(self, tokens, merges=None):
        self.tokens = tokens
        self.ids = {token: i for i, token in enumerate(tokens)}
        self.merges = merges
        self.ranks = {tuple(pair):i for i,pair in enumerate(merges or [])}
        self.cache = {}

    @classmethod
    def fit_bpe(cls, texts, vocabulary=3072):
        """Train character BPE with incremental weighted pair counts from scratch."""
        counts=Counter(piece for text in texts for piece in pieces(text))
        chars=sorted(set("".join(counts)) | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;-_()\n"))
        tokens=SPECIAL+chars
        sequences=[list(word) for word in counts]
        frequencies=list(counts.values())
        pair_counts=Counter()
        locations={}
        for i,seq in enumerate(sequences):
            for pair,count in Counter(zip(seq,seq[1:])).items():
                pair_counts[pair]+=count*frequencies[i]
                locations.setdefault(pair,set()).add(i)
        heap=[(-count,pair) for pair,count in pair_counts.items()]
        heapq.heapify(heap)
        merges=[]
        existing=set(tokens)
        while heap and len(tokens)<vocabulary:
            negative,pair=heapq.heappop(heap)
            if -negative!=pair_counts[pair] or pair_counts[pair]<2:
                continue
            merged="".join(pair)
            merges.append(list(pair))
            if merged not in existing:
                tokens.append(merged); existing.add(merged)
            changed=set()
            for i in list(locations.get(pair,())):
                old=sequences[i]
                before=Counter(zip(old,old[1:]))
                new=[]; j=0
                while j<len(old):
                    if j+1<len(old) and (old[j],old[j+1])==pair:
                        new.append(merged); j+=2
                    else:
                        new.append(old[j]); j+=1
                after=Counter(zip(new,new[1:]))
                for key in before.keys()|after.keys():
                    difference=after[key]-before[key]
                    if difference:
                        pair_counts[key]+=difference*frequencies[i]; changed.add(key)
                    if after[key]: locations.setdefault(key,set()).add(i)
                    else: locations.get(key,set()).discard(i)
                sequences[i]=new
            for key in changed:
                if pair_counts[key]>1: heapq.heappush(heap,(-pair_counts[key],key))
        return cls(tokens,merges)

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
            if self.merges is not None:
                cached=self.cache.get(piece)
                if cached is None:
                    parts=list(piece)
                    while len(parts)>1:
                        pairs=[(self.ranks.get((parts[i],parts[i+1]),float("inf")),i) for i in range(len(parts)-1)]
                        rank,index=min(pairs)
                        if rank==float("inf"): break
                        parts[index:index+2]=[parts[index]+parts[index+1]]
                    cached=[self.ids.get(part,UNKNOWN) for part in parts]
                    # Cache only public vocabulary entries, never arbitrary user words.
                    if piece in self.ids and len(self.cache)<30000: self.cache[piece]=cached
                result.extend(cached)
                continue
            if piece in self.ids:
                result.append(self.ids[piece])
            else:
                result.extend(self.ids.get(char, UNKNOWN) for char in piece)
        return result

    def decode(self, ids):
        return "".join(self.tokens[i] if i >= len(SPECIAL) else "�" if i == UNKNOWN else "" for i in ids)

    def save(self, path):
        data=self.tokens if self.merges is None else {"type":"bpe","tokens":self.tokens,"merges":self.merges}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["tokens"],data["merges"]) if isinstance(data,dict) else cls(data)


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
