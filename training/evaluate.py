"""Sealed test comparison. No checkpoint selection or training happens here."""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from neural.transformer import Transformer
from neural.tokenizer import prompt_tokens


def load_tests():
    examples=[json.loads(line) for line in (ROOT/"training-data/test.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen=[]
    for category in ("binding","correction","missing_fact","dialogue"):
        subset=[e for e in examples if e["category"]==category]
        random.Random(941).shuffle(subset)
        chosen.extend(subset[:60])
    return chosen


def answer_nll_per_byte(model,example):
    # Same raw text prefix for both models: token losses alone cannot compare tokenizers.
    target_text=example["answer"][:120]
    target=model.tokenizer.encode(target_text)
    limit=min(256,model.config["context"]-len(target))
    if limit<16 or not target: return None
    prompt=prompt_tokens(model.tokenizer,example["messages"],example.get("memory",""),limit=limit)
    logits,_,_=model.forward(prompt+target[:-1])
    logits=logits[len(prompt)-1:len(prompt)+len(target)-1].astype(np.float64)
    maximum=logits.max(axis=-1)
    logsum=maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=-1))
    nll=float((logsum-logits[np.arange(len(target)),target]).sum())
    return {"nll":nll,"bytes":len(target_text.encode("utf-8"))}


def evaluate(directory):
    model=Transformer(directory)
    rows=[]; groups=defaultdict(list); byte_nll=0.; byte_count=0; times=[]
    for e in load_tests():
        start=time.monotonic()
        output=model.generate(e["messages"],e.get("memory",""),max_tokens=128)
        times.append(time.monotonic()-start)
        expected=e["answer"].strip()
        row={"category":e["category"],"source":e["source"],"source_id":e.get("source_id"),"question":e["messages"][-1]["content"],
             "memory":e.get("memory",""),"history":e["messages"][:-1],"expected":expected,"actual":output["text"],"exact":output["text"]==expected}
        rows.append(row)
        if e["category"]=="dialogue":
            score=answer_nll_per_byte(model,e)
            if score:
                byte_nll+=score["nll"]; byte_count+=score["bytes"]
        else: groups[e["category"]].append(row["exact"])
    accuracy={category:{"correct":sum(values),"count":len(values),"accuracy":sum(values)/len(values)} for category,values in groups.items()}
    return {"config":model.config,"synthetic":accuracy,"synthetic_macro_accuracy":sum(v["accuracy"] for v in accuracy.values())/len(accuracy),
            "dialogue_nll_per_utf8_byte":byte_nll/byte_count if byte_count else None,"dialogue_scored_bytes":byte_count,
            "generation_seconds_median":float(np.median(times)),"generation_seconds_p95":float(np.quantile(times,.95)),"cases":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="neural/checkpoint")
    parser.add_argument("--output",default="training-output/baseline.json")
    args=parser.parse_args()
    result=evaluate(ROOT/args.model)
    path=ROOT/args.output; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2),flush=True)


if __name__=="__main__": main()
