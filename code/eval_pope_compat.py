"""Standard-metric evaluation on the POPE-compatible balanced subset (greedy decoding).
Reports accuracy, precision, recall, F1, yes-ratio for base model or a LoRA adapter.
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, re, argparse, random
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

HC = Path("/home/ubuntu/342/jinkwon/orthocampus/HalluCompass")
OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")
SNAP = "/home/ubuntu/.cache/huggingface/hub/datasets--anonymous80934--HalluCompass/snapshots/da2a24e1b7f0363a638942a84d825d3209bb49b9/images"

def find_image(fname):
    for sub in ["coco","nocaps","vizwiz","amber",""]:
        p=Path(SNAP)/sub/fname
        if p.exists(): return p
    h=list(Path(SNAP).rglob(fname)); return h[0] if h else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="")   # empty = base model
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--skeptical", action="store_true")
    a=ap.parse_args(); random.seed(0)
    rows=[json.loads(l) for l in (HC/"queries/pope_compat.jsonl").read_text().splitlines() if l.strip()]
    random.shuffle(rows); rows=rows[:a.n]
    path = a.adapter if a.adapter else a.model
    model, proc = FastVisionModel.from_pretrained(path, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in path.lower():
        # transformers>=4.47 llava-hf: explicit vision config needed for token expansion
        try:
            proc.patch_size = 14
            proc.vision_feature_select_strategy = "default"
            if hasattr(proc, "num_additional_image_tokens"):
                proc.num_additional_image_tokens = 1
        except Exception:
            pass

    FastVisionModel.for_inference(model); model.eval()
    try: proc.tokenizer.padding_side="left"
    except Exception: pass
    tp=fp=tn=fn=miss=0
    B=16
    def predict_batch(batch):
        imgs, texts = [], []
        for q in batch:
            ip=find_image(q["filename"])
            if ip is None: texts.append(None); imgs.append(None); continue
            im=Image.open(ip).convert("RGB"); im.thumbnail((512,512))
            qtext=q["query"]
            if a.skeptical:
                qtext="Be conservative. Only answer yes if you can clearly see the object; if unsure, answer no. "+qtext
            msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":qtext}]}]
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
    i=0
    while i < len(rows):
        batch=rows[i:i+B]
        try:
            preds=predict_batch(batch)
        except Exception as e:
            print("[batch failed, sequential fallback]", str(e)[:100], flush=True)
            preds=[]
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
        if i%320==0: print(f"  {i}/{len(rows)}", flush=True)
    n=tp+fp+tn+fn
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    res={"tag":a.tag,"n":n,"acc":round((tp+tn)/max(n,1),4),
         "precision":round(prec,4),"recall":round(rec,4),
         "f1":round(2*prec*rec/max(prec+rec,1e-9),4),
         "yes_ratio":round((tp+fp)/max(n,1),4),"unparsed":miss}
    print("[pope_compat]",res, flush=True)
    (OUT/f"pope_{a.tag}.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
