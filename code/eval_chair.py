"""CHAIR + utility on generated captions (base or adapter).
Ground truth per image = annotation truth[]. Object vocabulary = union of truth+hallu across dataset.
CHAIR_s = captions with >=1 hallucinated object / N ; CHAIR_i = hallucinated mentions / total mentions.
Also reports mean length and degenerate rate (utility). n=200 images across 4 sources.
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, glob, re, argparse, random
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

def build_vocab(anns):
    v=set()
    for a in anns:
        for o in a.get("truth",[])+a.get("hallu_pop",[])+a.get("hallu_adv",[]):
            o=o.strip().lower()
            if len(o)>=3 and o.replace(" ","").isalpha(): v.add(o)
    # sort longest-first for greedy longest-match
    return sorted(v, key=len, reverse=True)

def mentioned_objects(caption, vocab):
    c=" "+caption.lower()+" "; found=[]; used=[]
    for term in vocab:
        pat=r"(?<![a-z])"+re.escape(term)+r"(?![a-z])"
        m=re.search(pat,c)
        if m:
            span=(m.start(),m.end())
            if any(not(span[1]<=u[0] or span[0]>=u[1]) for u in used): continue  # overlaps longer match
            used.append(span); found.append(term)
    return found

def run(model, proc, anns, vocab):
    per=[]; halluc_caps=0; total_ment=0; halluc_ment=0
    for a in anns:
        ip=find_image(a["filename"])
        if ip is None: continue
        im=Image.open(ip).convert("RGB"); im.thumbnail((512,512))
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe this image in one sentence."}]}]
        text=proc.apply_chat_template(msg, add_generation_prompt=True)
        enc=proc(images=[im], text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=60, do_sample=False)
        cap=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        truth={t.strip().lower() for t in a.get("truth",[])}
        ment=mentioned_objects(cap, vocab)
        hall=[m for m in ment if m not in truth and not any(m in t or t in m for t in truth)]
        if hall: halluc_caps+=1
        total_ment+=len(ment); halluc_ment+=len(hall)
        per.append({"file":a["filename"],"caption":cap,"n_words":len(cap.split()),
                    "mentioned":len(ment),"halluc":hall})
    n=len(per)
    return {"n":n,
            "chair_s":round(halluc_caps/max(n,1),4),
            "chair_i":round(halluc_ment/max(total_ment,1),4),
            "mean_words":round(sum(p["n_words"] for p in per)/max(n,1),1),
            "degenerate_rate":round(sum(1 for p in per if p["n_words"]<4)/max(n,1),4),
            "samples":per[:8]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    a=ap.parse_args(); random.seed(0)
    anns=[json.load(open(f)) for f in glob.glob(str(HC/"annotations/*.json"))]
    vocab=build_vocab(anns)
    anns=[x for x in anns if x.get("truth")]
    random.shuffle(anns); anns=anns[:a.n]
    print(f"[chair:{a.tag}] {len(anns)} images, vocab={len(vocab)}", flush=True)
    path=a.adapter if a.adapter else a.model
    model, proc = FastVisionModel.from_pretrained(path, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in path.lower():
        try:
            proc.patch_size=14
            proc.vision_feature_select_strategy="default"
            if hasattr(proc, "num_additional_image_tokens"):
                proc.num_additional_image_tokens = 1
        except Exception: pass
    FastVisionModel.for_inference(model); model.eval()
    res=run(model, proc, anns, vocab)
    print(f"[chair:{a.tag}] "+", ".join(f"{k}={res[k]}" for k in ('n','chair_s','chair_i','mean_words','degenerate_rate')), flush=True)
    (OUT/f"chair_{a.tag}.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
