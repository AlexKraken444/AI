"""Reproducible Russian OASST1 import + authored context exercises.

Downloads public data, never local chat history. OASST1: Apache-2.0.
Prepared examples keep source message IDs for attribution and auditing.
"""
import gzip
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.corpus import corpus

DATA = ROOT / "training-data"
FILE = "2023-04-12_oasst_ready.messages.jsonl.gz"
REPO = "OpenAssistant/oasst1"


def read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Kraken-training/3.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def labels(row):
    value = row.get("labels") or {}
    if "name" in value and "value" in value:
        return dict(zip(value["name"], value["value"]))
    return {key: item.get("value", 0) if isinstance(item, dict) else item for key, item in value.items()}


def acceptable(row):
    if row.get("lang") != "ru" or row.get("deleted") or row.get("review_result") is not True:
        return False
    text = row.get("text", "")
    if not isinstance(text, str) or not 2 <= len(text) <= 2200:
        return False
    scores = labels(row)
    if row.get("role") == "assistant" and scores.get("quality", 1) < .6:
        return False
    if any(scores.get(key, 0) > .1 for key in ("spam", "pii", "hate_speech", "sexual_content")):
        return False
    if scores.get("violence",0) > .4 or scores.get("not_appropriate",0) > .3:
        return False
    if scores.get("toxicity", 0) > .2 or scores.get("lang_mismatch", 0) > .1:
        return False
    if re.search(r"https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+7|\+38)\s*\(?\d", text, re.I):
        return False
    return True


def oasst_examples(rows):
    by_id = {r["message_id"]: r for r in rows}
    examples = []
    rejected = Counter()
    for row in rows:
        if row.get("role") != "assistant" or row.get("lang") != "ru":
            continue
        if not acceptable(row) or row.get("rank") not in (0, 1, None):
            rejected["answer_quality_language_or_rank"] += 1
            continue
        chain, current, seen = [], row, set()
        while current and current["message_id"] not in seen:
            seen.add(current["message_id"])
            chain.append(current)
            parent = current.get("parent_id")
            current = by_id.get(parent) if parent else None
            if parent and current is None:
                chain = []
                break
        chain.reverse()
        if len(chain) < 2 or chain[0].get("parent_id") or any(not acceptable(r) for r in chain):
            rejected["incomplete_or_rejected_ancestor"] += 1
            continue
        if any(r["role"] != ("prompter" if i % 2 == 0 else "assistant") for i, r in enumerate(chain)):
            rejected["invalid_role_sequence"] += 1
            continue
        question_chain = [{"role": "user" if r["role"] == "prompter" else "assistant", "content": r["text"].strip()} for r in chain[:-1]]
        root_key = hashlib.sha256(normalize(chain[0]["text"]).encode()).hexdigest()
        examples.append({"messages":question_chain, "answer":row["text"].strip(), "memory":"", "group":"oasst:" + root_key,
                         "category":"dialogue", "source":"OpenAssistant/oasst1", "source_id":row["message_id"], "tree_id":row["message_tree_id"]})
    return examples, dict(rejected)


def context_exercises():
    rng = random.Random(20260922)
    names = ["Аня", "Борис", "Вера", "Глеб", "Даша", "Егор", "Женя", "Ира", "Кирилл", "Лена", "Маша", "Никита", "Оля", "Павел", "Рома", "Саша", "Таня", "Федя", "Юля", "Алекс"]
    cities = ["Москва", "Казань", "Омск", "Томск", "Самара", "Тула", "Пермь", "Уфа", "Сочи", "Псков", "Рязань", "Смоленск"]
    objects = ["чай", "кофе", "какао", "сок", "хлеб", "сыр", "рис", "шоколад", "мёд", "йогурт", "компот", "кефир"]
    result = []
    def add(messages, answer, group, category, memory=""):
        result.append({"messages":messages, "answer":answer, "memory":memory, "group":"context3:"+group, "category":category, "source":"authored-context-v3"})
    for _ in range(1400):
        a,b = rng.sample(names,2)
        x,y = rng.sample(objects,2)
        key = f"binding:{a}:{b}:{x}:{y}"
        facts = f"{a} любит {x}. {b} любит {y}."
        # Both questions and both orders stay in the same split.
        for person, obj in ((a,x),(b,y)):
            question=f"Что любит {person}?"
            add([{"role":"user","content":facts+" "+question}],f"{person} любит {obj}.",key,"binding")
            add([{"role":"user","content":f"{b} любит {y}. {a} любит {x}. "+question}],f"{person} любит {obj}.",key,"binding")
    for _ in range(900):
        old,new=rng.sample(cities,2)
        name=rng.choice(names)
        key=f"correction:{name}:{old}:{new}"
        fact=f"Имя: {name}. Город: {old}."
        history=[{"role":"user","content":fact},{"role":"assistant","content":"Учту."},{"role":"user","content":f"Исправление: мой город теперь {new}. Какой город актуален?"}]
        add(history,f"Актуальный город — {new}.",key,"correction")
        add([{"role":"user","content":f"Раньше мой город был {old}, теперь — {new}. Назови актуальный город."}],f"Актуальный город — {new}.",key,"correction")
        add([{"role":"user","content":f"Старую запись нужно исправить: мой город теперь {new}. Какой город актуален?"}],f"Актуальный город — {new}.",key,"correction",fact)
    for _ in range(500):
        name=rng.choice(names); obj=rng.choice(objects); other=rng.choice(cities)
        key=f"unknown:{name}:{obj}:{other}"
        add([{"role":"user","content":f"{name} любит {obj}. В каком городе живёт {name}?"}],"В сообщении не указан город. Недостаточно данных для ответа.",key,"missing_fact")
        add([{"role":"user","content":f"Город: {other}. Как зовут человека?"}],"Имя не указано. Недостаточно данных для ответа.",key,"missing_fact")
    return result


def split_examples(examples):
    splits = {"train":[],"validation":[],"test":[]}
    unique = {}
    for example in examples:
        signature = json.dumps([[(m["role"],normalize(m["content"])) for m in example["messages"]],normalize(example.get("memory",""))], ensure_ascii=False)
        if signature in unique:
            continue
        unique[signature]=example
        bucket=int(hashlib.sha256(example["group"].encode()).hexdigest()[:8],16)%20
        split="test" if bucket == 0 else "validation" if bucket == 1 else "train"
        splits[split].append(example)
    return splits


def main():
    DATA.mkdir(exist_ok=True)
    manifest_path=DATA/"download.json"
    if manifest_path.exists():
        download=json.loads(manifest_path.read_text(encoding="utf-8"))
    elif (ROOT/"training/data_manifest.json").exists():
        download=json.loads((ROOT/"training/data_manifest.json").read_text(encoding="utf-8"))["source"]
    else:
        info=json.loads(read_url(f"https://huggingface.co/api/datasets/{REPO}"))
        download={"repository":REPO,"revision":info["sha"],"filename":FILE,"license":"Apache-2.0"}
        manifest_path.write_text(json.dumps(download,indent=2),encoding="utf-8")
    archive=DATA/FILE
    if not archive.exists():
        print("Downloading OASST1 public messages at pinned revision "+download["revision"],flush=True)
        body=read_url(f"https://huggingface.co/datasets/{REPO}/resolve/{download['revision']}/{FILE}")
        temporary=archive.with_suffix(".tmp")
        temporary.write_bytes(body)
        temporary.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if download.get("sha256") and download["sha256"] != digest:
        raise ValueError("Downloaded corpus hash mismatch")
    download["sha256"]=digest
    manifest_path.write_text(json.dumps(download,indent=2),encoding="utf-8")
    with gzip.open(archive,"rt",encoding="utf-8") as source:
        rows=[json.loads(line) for line in source]
    dialogue,rejected=oasst_examples(rows)
    old_train,old_validation=corpus()
    legacy=[dict(e,source="authored-v2",category="legacy") for e in old_train+old_validation]
    splits=split_examples(legacy+context_exercises()+dialogue)
    report={"source":download,"raw_messages":len(rows),"accepted_oasst_examples":len(dialogue),"rejected":rejected,"splits":{}}
    for name,examples in splits.items():
        path=DATA/(name+".jsonl")
        path.write_text("".join(json.dumps(e,ensure_ascii=False)+"\n" for e in examples),encoding="utf-8")
        report["splits"][name]={"examples":len(examples),"by_source":dict(Counter(e["source"] for e in examples)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT/"training/data_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
