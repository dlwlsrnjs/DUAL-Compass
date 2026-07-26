"""Run any unsloth VLM over id_{rand,pop,adv} + optional extra queries -> DUAL streams."""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
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
    h = list(Path(SNAP).rglob(fname)); return h[0] if h else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--with_extra", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, proc = FastVisionModel.from_pretrained(a.model, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in a.model.lower():
        # transformers>=4.47 llava-hf: explicit vision config needed for token expansion
        try:
            proc.patch_size = 14
            proc.vision_feature_select_strategy = "default"
            if hasattr(proc, "num_additional_image_tokens"):
                proc.num_additional_image_tokens = 1
        except Exception:
            pass

    FastVisionModel.for_inference(model); model.eval()
    queries = []
    for split in ["rand","pop","adv"]:
        for l in (HC/"queries"/f"id_{split}.jsonl").read_text().splitlines():
            if l.strip():
                d = json.loads(l); d["_split"] = split; queries.append(d)
    if a.with_extra and (OUT/"extra_queries.jsonl").exists():
        for l in (OUT/"extra_queries.jsonl").read_text().splitlines():
            if l.strip():
                d = json.loads(l); d["_split"] = "extra"; queries.append(d)
    if a.limit: queries = queries[:a.limit]
    print(f"[infer:{a.tag}] {len(queries)} queries (batched)", flush=True)
    try: proc.tokenizer.padding_side = "left"
    except Exception: pass
    recs = []; fp=fn=tp=tn=0
    B = int(os.environ.get("EXP_BATCH", "16"))
    def gen_batch(qs):
        imgs, texts, keep = [], [], []
        for j,q in enumerate(qs):
            ip = find_image(q["filename"])
            if ip is None: continue
            im = Image.open(ip).convert("RGB"); im.thumbnail((512,512))
            msg = [{"role":"user","content":[{"type":"image"},{"type":"text","text":q["query"]}]}]
            imgs.append(im); keep.append(j)
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True))
        preds = [None]*len(qs)
        if not keep: return preds
        try:
            enc = proc(images=imgs, text=texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=4, do_sample=False)
            outs = proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[batch failed -> sequential]", str(e)[:90], flush=True)
            outs = []
            for im, tx in zip(imgs, texts):
                enc = proc(images=[im], text=tx, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=4, do_sample=False)
                outs.append(proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        for j,g in zip(keep, outs):
            preds[j] = "yes" if re.search(r"\byes\b",g,re.I) else ("no" if re.search(r"\bno\b",g,re.I) else "?")
        return preds
    bi = 0
    while bi < len(queries):
        qbatch = queries[bi:bi+B]
        pbatch = gen_batch(qbatch)
        for q, pred in zip(qbatch, pbatch):
            if pred is None or pred == "?": continue
            label = q["label"]
            stream = ("unlearn" if (label=="no" and pred=="yes") else
                      "learn" if (label=="yes" and pred=="no") else "retain")
            if stream=="unlearn": fp+=1
            elif stream=="learn": fn+=1
            elif label=="yes": tp+=1
            else: tn+=1
            recs.append({"image":q["filename"],"object":q["object"],"query":q["query"],
                         "split":q["_split"],"label":label,"model_pred":pred,"stream":stream,
                         "gold_answer":"Yes." if label=="yes" else "No."})
        bi += B
        if bi % 512 == 0: print(f"  {bi}/{len(queries)} u={fp} l={fn} r={tp+tn}", flush=True)
    (OUT/f"dual_full_{a.tag}.jsonl").write_text("\n".join(json.dumps(r) for r in recs)+"\n")
    s=(fp-fn)/max(fp+fn,1)
    print(f"[done:{a.tag}] N={len(recs)} unlearn={fp} learn={fn} retain={tp+tn} s_beh={s:.3f}", flush=True)

if __name__=="__main__":
    main()
