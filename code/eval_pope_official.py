"""Official POPE benchmark (Li et al. 2023) on COCO val2014 — TRUE external evaluation.
Reads official coco_pope_{random,popular,adversarial}.json and local COCO images.
Reports per-split and overall acc/precision/recall/F1/yes-ratio + FP/FN and signed score
s=(FP-FN)/(FP+FN). Base model or LoRA adapter.
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

def load_split(sp, per_split=0):
    import random as _r
    rows=[json.loads(l) for l in (POPE/f"coco_pope_{sp}.json").read_text().splitlines() if l.strip()]
    if per_split and per_split < len(rows):
        _r.Random(0).shuffle(rows); rows=rows[:per_split]
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per_split", type=int, default=500)
    a=ap.parse_args()
    path=a.adapter if a.adapter else a.model
    model, proc = FastVisionModel.from_pretrained(path, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in path.lower():
        try:
            proc.patch_size=14; proc.vision_feature_select_strategy="default"
            if hasattr(proc,"num_additional_image_tokens"): proc.num_additional_image_tokens=1
        except Exception: pass
    FastVisionModel.for_inference(model); model.eval()
    try: proc.tokenizer.padding_side="left"
    except Exception: pass

    B=16
    _imcache={}
    def get_img(name):
        if name not in _imcache:
            ip=IMG/name
            if not ip.exists(): _imcache[name]=None
            else:
                im=Image.open(ip).convert("RGB"); im.thumbnail((512,512)); _imcache[name]=im
        return _imcache[name]
    def predict_batch(batch):
        imgs,texts=[],[]
        for q in batch:
            im=get_img(q["image"])
            if im is None: texts.append(None); imgs.append(None); continue
            msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q["text"]}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True)); imgs.append(im)
        keep=[j for j,t in enumerate(texts) if t is not None]
        preds=[None]*len(batch)
        if not keep: return preds
        enc=proc(images=[imgs[j] for j in keep], text=[texts[j] for j in keep],
                 return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=4, do_sample=False)
        outs=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j,g in zip(keep,outs):
            preds[j]="yes" if re.search(r"\byes\b",g,re.I) else ("no" if re.search(r"\bno\b",g,re.I) else "?")
        return preds

    all_res={"tag":a.tag,"model":path}
    agg=dict(tp=0,fp=0,tn=0,fn=0,miss=0)
    for sp in ["random","popular","adversarial"]:
        rows=load_split(sp, a.per_split)
        tp=fp=tn=fn=miss=0; i=0
        while i<len(rows):
            batch=rows[i:i+B]
            try: preds=predict_batch(batch)
            except Exception as e:
                print("[batch fail seq]",str(e)[:80],flush=True); preds=[]
                for q in batch:
                    try: preds.extend(predict_batch([q]))
                    except Exception: preds.append(None)
            for q,pred in zip(batch,preds):
                if pred is None or pred=="?": miss+=1; continue
                lab=q["label"]
                if lab=="yes" and pred=="yes": tp+=1
                elif lab=="yes": fn+=1
                elif pred=="yes": fp+=1
                else: tn+=1
            i+=B
            if i % (B*30)==0: print(f"  [{a.tag}:{sp}] {i}/{len(rows)}",flush=True)
        n=tp+fp+tn+fn; prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        signed=(fp-fn)/max(fp+fn,1)
        r={"n":n,"acc":round((tp+tn)/max(n,1),4),"precision":round(prec,4),"recall":round(rec,4),
           "f1":round(2*prec*rec/max(prec+rec,1e-9),4),"yes_ratio":round((tp+fp)/max(n,1),4),
           "fp":fp,"fn":fn,"signed":round(signed,4),"unparsed":miss}
        all_res[sp]=r
        for k in ["tp","fp","tn","fn","miss"]: agg[k]+=locals()[k]
        print(f"[{a.tag}:{sp}] acc={r['acc']} f1={r['f1']} yr={r['yes_ratio']} signed={r['signed']} fp={fp} fn={fn}",flush=True)
    tp,fp,tn,fn=agg["tp"],agg["fp"],agg["tn"],agg["fn"]; n=tp+fp+tn+fn
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    all_res["overall"]={"n":n,"acc":round((tp+tn)/max(n,1),4),"precision":round(prec,4),"recall":round(rec,4),
        "f1":round(2*prec*rec/max(prec+rec,1e-9),4),"yes_ratio":round((tp+fp)/max(n,1),4),
        "fp":fp,"fn":fn,"signed":round((fp-fn)/max(fp+fn,1),4),"unparsed":agg["miss"]}
    o=all_res["overall"]
    print(f"[{a.tag}:OVERALL] acc={o['acc']} f1={o['f1']} yr={o['yes_ratio']} signed={o['signed']}",flush=True)
    (OUT/f"popeofficial_{a.tag}.json").write_text(json.dumps(all_res,indent=2))
    print("[wrote]",OUT/f"popeofficial_{a.tag}.json",flush=True)

if __name__=="__main__":
    main()
