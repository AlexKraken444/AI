"""Train and select a candidate checkpoint without overwriting the deployed model.

Validation selects the checkpoint; sealed test examples are used only by evaluate.py.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/".training-libs"))
import numpy as np
import torch
from torch.nn import functional as F
from train_transformer import LanguageModel
from neural.tokenizer import Tokenizer, END, prompt_tokens


def read_examples(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def encode_examples(examples, tokenizer, prompt_limit=256, answer_limit=224):
    encoded=[]; rejected=Counter()
    for e in examples:
        answer=tokenizer.encode(e["answer"])
        # Never append an end token to an arbitrarily cut-off answer.
        if len(answer)>answer_limit:
            rejected[e["source"]]+=1
            continue
        prompt=prompt_tokens(tokenizer,e["messages"],e.get("memory",""),limit=prompt_limit)
        ids=prompt+answer+[END]
        labels=[-100]*(len(prompt)-1)+answer+[END]
        encoded.append({"ids":ids[:-1],"labels":labels,"category":e["category"],"source":e["source"]})
    return encoded,dict(rejected)


def batch(examples,indices):
    rows=[examples[i] for i in indices]
    length=max(len(row["ids"]) for row in rows)
    x=torch.zeros((len(rows),length),dtype=torch.long)
    y=torch.full((len(rows),length),-100,dtype=torch.long)
    for i,row in enumerate(rows):
        x[i,:len(row["ids"])]=torch.tensor(row["ids"])
        y[i,:len(row["labels"])]=torch.tensor(row["labels"])
    return x,y


def validation_score(model,examples,vocabulary):
    sums=Counter(); counts=Counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0,len(examples),12):
            indices=list(range(offset,min(offset+12,len(examples))))
            x,y=batch(examples,indices)
            logits=model(x)
            losses=F.cross_entropy(logits.reshape(-1,vocabulary),y.reshape(-1),reduction="none").reshape_as(y)
            per_example=losses.sum(dim=1)/(y!=-100).sum(dim=1)
            for index,loss in zip(indices,per_example.tolist()):
                group=examples[index]["category"]
                sums[group]+=loss; counts[group]+=1
    group_losses={group:sums[group]/counts[group] for group in sums}
    return sum(group_losses.values())/len(group_losses),group_losses


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=3200)
    parser.add_argument("--batch",type=int,default=12)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output",default="training-output/context3")
    args=parser.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(43)
    rng=np.random.default_rng(43)
    train=read_examples(ROOT/"training-data/train.jsonl")
    validation=read_examples(ROOT/"training-data/validation.jsonl")
    output=ROOT/args.output; output.mkdir(parents=True,exist_ok=True)
    print("Fitting BPE on training text only",flush=True)
    tokenizer=Tokenizer.fit_bpe([text for e in train for text in [e["answer"],e.get("memory","")]+[m["content"] for m in e["messages"]]],vocabulary=3072)
    tokenizer.save(output/"tokenizer.json")
    config={"version":"kraken-context-3","width":160,"heads":4,"layers":3,"context":512,"vocabulary":len(tokenizer.tokens),"seed":43,"tokenizer":"bpe"}
    model=LanguageModel(config)
    config["parameters"]=sum(p.numel() for p in model.parameters())
    (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    training,rejected_train=encode_examples(train,tokenizer)
    testing,rejected_validation=encode_examples(validation,tokenizer)
    print(f"{config['parameters']:,} parameters; {len(training)} train; {len(testing)} validation; rejected long answers {rejected_train}",flush=True)
    pools={}
    for i,e in enumerate(training): pools.setdefault(e["category"],[]).append(i)
    # Language learning is not swamped by thousands of short synthetic examples.
    categories=list(pools)
    weights=np.array([4 if group=="dialogue" else 2 if group=="legacy" else 1 for group in categories],dtype=float)
    weights/=weights.sum()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    monitor=[]
    for category in sorted({e["category"] for e in testing}):
        subset=[e for e in testing if e["category"]==category]
        random.Random(43).shuffle(subset)
        monitor.extend(subset[:60])
    best=float("inf"); best_step=0; history=[]; start=time.monotonic()
    for step in range(1,args.steps+1):
        model.train()
        category=rng.choice(categories,p=weights)
        pool=pools[category]
        # Similar lengths reduce padding; drawing candidates does not alter dataset splits.
        candidate=rng.choice(pool,size=args.batch*4)
        candidate=sorted(candidate,key=lambda i:len(training[i]["ids"]))
        begin=int(rng.integers(0,len(candidate)-args.batch+1))
        x,y=batch(training,candidate[begin:begin+args.batch])
        optimizer.param_groups[0]["lr"]=.001*min(1,step/100)*(.15+.85*(1+math.cos(math.pi*step/args.steps))/2)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        losses=F.cross_entropy(logits.reshape(-1,len(tokenizer.tokens)),y.reshape(-1),reduction="none").reshape_as(y)
        loss=(losses.sum(dim=1)/(y!=-100).sum(dim=1)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        if step%100==0:
            print(f"step {step}/{args.steps} train={float(loss.detach()):.3f} group={category} elapsed={time.monotonic()-start:.1f}s",flush=True)
        if step%400==0 or step==args.steps:
            score,groups=validation_score(model,monitor,len(tokenizer.tokens))
            history.append({"step":step,"score":score,"groups":groups})
            if score<best:
                best=score; best_step=step
                temp=output/"candidate.npz"
                np.savez_compressed(temp,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
                temp.replace(output/"model.npz")
            print(f"validation={score:.3f}; best step={best_step}",flush=True)
    # Evaluate full validation only after checkpoint selection; test remains untouched.
    with np.load(output/"model.npz",allow_pickle=False) as archive:
        model.load_state_dict({k:torch.from_numpy(archive[k]) for k in archive.files})
    full_score,full_groups=validation_score(model,testing,len(tokenizer.tokens))
    manifest=json.loads((ROOT/"training/data_manifest.json").read_text(encoding="utf-8"))
    metrics={"steps":args.steps,"selected_step":best_step,"seed":43,"parameters":config["parameters"],
             "training_examples":len(training),"validation_examples":len(testing),"training_by_source":dict(Counter(e["source"] for e in training)),
             "rejected_long_answers_train":rejected_train,"rejected_long_answers_validation":rejected_validation,
             "validation_macro_example_loss":full_score,"validation_by_category":full_groups,"selection_history":history,
             "elapsed_seconds":round(time.monotonic()-start,2),"dataset_hashes":{name:item["sha256"] for name,item in manifest["splits"].items()},
             "note":"Small dataset; losses are not accuracy. Test set was not used in training or checkpoint selection."}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    from neural.transformer import Transformer
    runtime=Transformer(output)
    ids=prompt_tokens(tokenizer,[{"role":"user","content":"Что такое Python?"}])
    model.eval()
    with torch.no_grad(): reference=model(torch.tensor([ids]))[0].numpy()
    actual,_,_=runtime.forward(ids)
    np.testing.assert_allclose(actual,reference,atol=5e-4,rtol=5e-4)
    print("Candidate ready; NumPy/PyTorch parity passed",flush=True)


if __name__=="__main__": main()
