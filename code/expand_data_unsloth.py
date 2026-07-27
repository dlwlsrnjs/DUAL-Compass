"""Expand DUAL data: run Qwen2-VL-2B (unsloth) over the FULL id query set (4003)
to get a large error pool, then split into unlearn/learn/retain streams.

label=no & pred=yes -> unlearn (yes-bias / over-claim)
label=yes & pred=no -> learn   (no-bias / missed)
correct             -> retain  (anchor)
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("USE_FLAX","0")
import json, re, argparse
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

HC = Path("/home/ubuntu/342/jinkwon/orthocampus/HalluCompass")
OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")
SNAP = "/home/ubuntu/.cache/huggingface/hub/datasets--anonymous80934--HalluCompass/snapshots/da2a24e1b7f0363a638942a84d825d3209bb49b9/images"

def find_image(fname):
    for sub in ["coco","nocaps","vizwiz","amber",""]:
        p = Path(SNAP)/sub/fname
        if p.exists(): return p
    h=list(Path(SNAP).rglob(fname)); return h[0] if h else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--tag", default="qwen2vl2b")
    ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args()

    model, proc = FastVisionModel.from_pretrained(a.model, load_in_4bit=False, dtype=torch.bfloat16)
    FastVisionModel.for_inference(model)
    model.eval()

    queries=[]
    for split in ["rand","pop","adv"]:
        for l in (HC/"queries"/f"id_{split}.jsonl").read_text().splitlines():
            if l.strip():
                d=json.loads(l); d["_split"]=split; queries.append(d)
    if a.limit: queries=queries[:a.limit]
    print(f"[infer] {len(queries)} id queries", flush=True)

    recs=[]; fp=fn=tp=tn=miss=0
    for i,q in enumerate(queries):
        ip=find_image(q["filename"])
        if ip is None: miss+=1; continue
        image=Image.open(ip).convert("RGB"); image.thumbnail((512,512))
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q["query"]}]}]
        text=proc.apply_chat_template(msg, add_generation_prompt=True)
        enc=proc(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**enc, max_new_tokens=4, do_sample=False)
        gen=proc.batch_decode(out[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pred="yes" if re.search(r"\byes\b", gen, re.I) else ("no" if re.search(r"\bno\b", gen, re.I) else "?")
        label=q["label"]
        if pred=="?": continue
        stream = ("unlearn" if (label=="no" and pred=="yes")
                  else "learn" if (label=="yes" and pred=="no")
                  else "retain" if label==pred else None)
        if stream is None: continue
        if stream=="unlearn": fp+=1
        elif stream=="learn": fn+=1
        elif label=="yes": tp+=1
        else: tn+=1
        recs.append({"image":q["filename"],"object":q["object"],"query":q["query"],
                     "split":q["_split"],"label":label,"model_pred":pred,
                     "bias_direction":q.get("bias_direction",""),
                     "stream":stream,"gold_answer":"Yes." if label=="yes" else "No."})
        if (i+1)%500==0: print(f"  {i+1}/{len(queries)} unlearn={fp} learn={fn} retain={tp+tn}", flush=True)

    outp=OUT/f"dual_full_{a.tag}.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in recs)+"\n")
    s=(fp-fn)/max(fp+fn,1)
    print(f"[done] N={len(recs)} unlearn={fp} learn={fn} retain={tp+tn} miss_img={miss} "
          f"s_beh={s:.3f} fp_rate={fp/max(fp+tn,1):.3f} fn_rate={fn/max(fn+tp,1):.3f}", flush=True)
    print(f"[wrote] {outp}", flush=True)

if __name__=="__main__":
    main()
