"""Build a DASH-derived direction stream from the DASH-B TRAIN split.
Eval protocol integrity: eval_dashb.py uses shuffle(seed 0).head(1000); we use rows[1000:] only.
Diagnose the subject model on those rows (batched), emit dual-format records with
absolute image paths, and merge with the suite streams into a _mix stream file.
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, re, io, argparse
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")
IMGDIR = OUT/"dashb_train_imgs"; IMGDIR.mkdir(exist_ok=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--suite_tag", default="qwen2vl2b_ext")
    ap.add_argument("--out_tag", default="qwen2vl2b_mix")
    a=ap.parse_args()
    import pandas as pd
    from huggingface_hub import snapshot_download
    d=snapshot_download("YanNeu/DASH-B", repo_type="dataset", allow_patterns=["data/*"])
    df=pd.concat([pd.read_parquet(f) for f in sorted(Path(d).glob("data/*.parquet"))], ignore_index=True)
    df=df.sample(frac=1.0, random_state=0)
    train=df.iloc[1000:]              # rows 1000+ = train split; first 1000 reserved for eval
    print(f"[dashb-train] {len(train)} rows (eval-first-1000 untouched)", flush=True)

    model, proc = FastVisionModel.from_pretrained(a.model, load_in_4bit=False, dtype=torch.bfloat16)
    FastVisionModel.for_inference(model); model.eval()
    try: proc.tokenizer.padding_side="left"
    except Exception: pass

    recs=[]; fp=fn=tp=tn=0; B=16
    buf=[]
    def flush():
        nonlocal fp,fn,tp,tn,buf
        if not buf: return
        imgs=[b[0] for b in buf]; texts=[b[1] for b in buf]
        enc=proc(images=imgs, text=texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=4, do_sample=False)
        outs=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (img,txt,q,lab,pth),g in zip(buf,outs):
            pred="yes" if re.search(r"\byes\b",g,re.I) else ("no" if re.search(r"\bno\b",g,re.I) else "?")
            if pred=="?": continue
            stream=("unlearn" if (lab=="no" and pred=="yes") else
                    "learn" if (lab=="yes" and pred=="no") else "retain")
            if stream=="unlearn": fp+=1
            elif stream=="learn": fn+=1
            elif lab=="yes": tp+=1
            else: tn+=1
            recs.append({"image":str(pth),"object":"","query":q,"split":"extra",
                         "label":lab,"model_pred":pred,"stream":stream,
                         "gold_answer":"Yes." if lab=="yes" else "No.","source":"dashb"})
        buf=[]
    for i,(_,row) in enumerate(train.iterrows()):
        img=row.get("image")
        if isinstance(img,dict) and "bytes" in img: img=img["bytes"]
        try: im=Image.open(io.BytesIO(img)).convert("RGB")
        except Exception: continue
        im.thumbnail((512,512))
        pth=IMGDIR/f"dashb_{i}.jpg"
        if not pth.exists(): im.save(pth, quality=90)
        lab=str(row.get("answer") or row.get("label")).strip().lower()
        lab="yes" if lab in ("1","true","yes","present") else ("no" if lab in ("0","false","no","absent") else "?")
        if lab=="?": continue
        obj=row.get("object") or ""
        q=row.get("question") or f'Is there a {obj} in this image? Answer with only "yes" or "no".'
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
        buf.append((im, proc.apply_chat_template(msg, add_generation_prompt=True), q, lab, pth))
        if len(buf)>=B: flush()
        if (i+1)%300==0: print(f"  {i+1}/{len(train)} u={fp} l={fn} r={tp+tn}", flush=True)
    flush()
    s=(fp-fn)/max(fp+fn,1)
    print(f"[dashb-diag] unlearn={fp} learn={fn} retain={tp+tn} s_beh={s:+.3f}", flush=True)

    suite=[json.loads(l) for l in (OUT/f"dual_full_{a.suite_tag}.jsonl").read_text().splitlines() if l.strip()]
    mixed=suite+recs
    (OUT/f"dual_full_{a.out_tag}.jsonl").write_text("\n".join(json.dumps(r) for r in mixed)+"\n")
    print(f"[mix] suite={len(suite)} + dashb={len(recs)} -> dual_full_{a.out_tag}.jsonl", flush=True)

if __name__=="__main__":
    main()
