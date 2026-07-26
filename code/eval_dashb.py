"""External transfer evaluation on DASH-B (Augustin et al., ICCV'25) — TNR/TPR/acc.
Loads YanNeu/DASH-B parquet from HF (images embedded).
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, re, io, argparse, random
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a=ap.parse_args(); random.seed(0)
    import pandas as pd
    from huggingface_hub import snapshot_download
    dpath = snapshot_download("YanNeu/DASH-B", repo_type="dataset", allow_patterns=["data/*"])
    files = sorted(Path(dpath).glob("data/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("[dashb] columns:", list(df.columns), "rows:", len(df), flush=True)
    df = df.sample(frac=1.0, random_state=0).head(a.n)
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
    pend_imgs, pend_texts, pend_labs = [], [], []
    B=16
    def flush():
        nonlocal tp,fp,tn,fn,miss,pend_imgs,pend_texts,pend_labs
        if not pend_imgs: return
        try:
            enc=proc(images=pend_imgs, text=pend_texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                o=model.generate(**enc, max_new_tokens=4, do_sample=False)
            outs=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[batch failed, sequential fallback]", str(e)[:100], flush=True)
            outs=[]
            for im,tx in zip(pend_imgs,pend_texts):
                try:
                    enc=proc(images=[im], text=tx, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        o=model.generate(**enc, max_new_tokens=4, do_sample=False)
                    outs.append(proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)[0])
                except Exception: outs.append("")
        for g,lab in zip(outs,pend_labs):
            pred="yes" if re.search(r"\byes\b",g,re.I) else ("no" if re.search(r"\bno\b",g,re.I) else "?")
            if pred=="?": miss+=1; continue
            if lab=="yes" and pred=="yes": tp+=1
            elif lab=="yes": fn+=1
            elif pred=="yes": fp+=1
            else: tn+=1
        pend_imgs, pend_texts, pend_labs = [], [], []
    for i,(_,row) in enumerate(df.iterrows()):
        # robust field detection
        img = row.get("image") or row.get("img")
        if isinstance(img, dict) and "bytes" in img: img = img["bytes"]
        try:
            image=Image.open(io.BytesIO(img)).convert("RGB") if isinstance(img,(bytes,bytearray)) else Image.open(img).convert("RGB")
        except Exception: miss+=1; continue
        image.thumbnail((512,512))
        obj = row.get("object") or row.get("object_name") or row.get("label_name")
        lab = row.get("label")
        if lab is None: lab = row.get("answer")
        lab = str(lab).strip().lower()
        if lab in ("1","true","yes","present"): lab="yes"
        elif lab in ("0","false","no","absent"): lab="no"
        q = row.get("question") or f'Is there a {obj} in this image? Answer with only "yes" or "no".'
        if lab not in ("yes","no"): miss+=1; continue
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
        pend_imgs.append(image); pend_labs.append(lab)
        pend_texts.append(proc.apply_chat_template(msg, add_generation_prompt=True))
        if len(pend_imgs)>=B: flush()
        if (i+1)%200==0: print(f"  {i+1}/{len(df)}", flush=True)
    flush()
    res={"tag":a.tag,"n":tp+fp+tn+fn,"acc":round((tp+tn)/max(tp+fp+tn+fn,1),4),
         "tnr":round(tn/max(tn+fp,1),4),"tpr":round(tp/max(tp+fn,1),4),
         "yes_ratio":round((tp+fp)/max(tp+fp+tn+fn,1),4),"skipped":miss}
    print("[dashb]",res, flush=True)
    (OUT/f"dashb_{a.tag}.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
