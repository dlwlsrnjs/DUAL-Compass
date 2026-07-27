"""Constructive POPE-COCO: diagnose a model ON COCO and build direction streams.
Splits the 500 official-POPE COCO images into train (first 300) / eval (last 200),
disjoint by image. Diagnoses the model (greedy yes/no) on the TRAIN questions, assigns
each to unlearn/learn/retain, and writes a dual_full_{tag}.jsonl consumable by
train_dual_loop.py. EVAL-image questions are written with split='adv' for the internal
held-out log-likelihood check; the headline greedy eval uses eval_pope_official on the
held-out subset separately.
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, re, argparse
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

POPE = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/pope_official")
IMG  = POPE/"images"
OUT  = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")

def load_all():
    rows=[]
    for sp in ["random","popular","adversarial"]:
        for l in (POPE/f"coco_pope_{sp}.json").read_text().splitlines():
            if l.strip():
                d=json.loads(l); d["pope_split"]=sp; rows.append(d)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--tag", required=True)      # e.g. cocopope2b
    ap.add_argument("--n_train_img", type=int, default=300)
    a=ap.parse_args()
    rows=load_all()
    imgs=sorted({r["image"] for r in rows})
    train_imgs=set(imgs[:a.n_train_img]); eval_imgs=set(imgs[a.n_train_img:])
    print(f"[split] {len(train_imgs)} train imgs / {len(eval_imgs)} eval imgs", flush=True)

    model, proc = FastVisionModel.from_pretrained(a.model, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in a.model.lower():
        try:
            proc.patch_size=14; proc.vision_feature_select_strategy="default"
            if hasattr(proc,"num_additional_image_tokens"): proc.num_additional_image_tokens=1
        except Exception: pass
    FastVisionModel.for_inference(model); model.eval()
    try: proc.tokenizer.padding_side="left"
    except Exception: pass

    _cache={}
    def img_of(name):
        if name not in _cache:
            p=IMG/name; im=Image.open(p).convert("RGB"); im.thumbnail((512,512)); _cache[name]=im
        return _cache[name]
    def predict(batch):
        texts=[]; ims=[]
        for r in batch:
            msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":r["text"]}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True)); ims.append(img_of(r["image"]))
        enc=proc(images=ims, text=texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=4, do_sample=False)
        outs=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return ["yes" if re.search(r"\byes\b",g,re.I) else ("no" if re.search(r"\bno\b",g,re.I) else "?") for g in outs]

    def rec(r, split, stream=None):
        lab=r["label"]
        return {"image":str(IMG/r["image"]), "query":r["text"], "label":lab,
                "gold_answer":"Yes." if lab=="yes" else "No.",
                "stream":stream or "retain", "split":split, "pope_split":r["pope_split"]}

    out=[]; B=16
    train_rows=[r for r in rows if r["image"] in train_imgs]
    cU=cL=cR=0; i=0
    while i<len(train_rows):
        b=train_rows[i:i+B]
        try: preds=predict(b)
        except Exception as e:
            print("[batch fail]",str(e)[:80],flush=True); preds=["?"]*len(b)
        for r,p in zip(b,preds):
            lab=r["label"]
            if lab=="no" and p=="yes": st="unlearn"; cU+=1
            elif lab=="yes" and p=="no": st="learn"; cL+=1
            else: st="retain"; cR+=1
            # distribute train records across rand/pop so probe/train draw from them
            split = "rand" if (i//B)%2==0 else "pop"
            out.append(rec(r, split, st))
        i+=B
        if i%320==0: print(f"  diag {i}/{len(train_rows)}",flush=True)
    # eval-image questions -> split 'adv' for internal held-out check
    for r in rows:
        if r["image"] in eval_imgs:
            out.append(rec(r, "adv"))
    s=(cU-cL)/max(cU+cL,1)
    print(f"[diagnosis] train streams: unlearn={cU} learn={cL} retain={cR}  s_beh={s:+.3f}", flush=True)
    (OUT/f"dual_full_{a.tag}.jsonl").write_text("\n".join(json.dumps(x) for x in out))
    print(f"[wrote] {OUT}/dual_full_{a.tag}.jsonl ({len(out)} records)", flush=True)

if __name__=="__main__":
    main()
