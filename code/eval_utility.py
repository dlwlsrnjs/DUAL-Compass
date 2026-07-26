"""Utility check: base vs trained adapter.
- generate a caption ('Describe this image in one sentence.') for N images
- hallucination: mentions of known-absent objects (hallu_pop/hallu_adv)
- fluency proxy: mean caption length in words; degenerate if <4 words
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, glob, argparse, random
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

def run(model, proc, anns):
    res=[]
    for an in anns:
        ip=find_image(an["filename"])
        if ip is None: continue
        image=Image.open(ip).convert("RGB"); image.thumbnail((512,512))
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe this image in one sentence."}]}]
        text=proc.apply_chat_template(msg, add_generation_prompt=True)
        enc=proc(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=60, do_sample=False)
        cap=proc.batch_decode(o[:,enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        absent=[x.lower() for x in an.get("hallu_pop",[])+an.get("hallu_adv",[])]
        halluc=[x for x in absent if x in cap.lower()]
        res.append({"file":an["filename"],"caption":cap,"halluc_mentions":halluc,"n_words":len(cap.split())})
    n=len(res)
    return {"n":n,
            "halluc_rate":round(sum(1 for r in res if r["halluc_mentions"])/max(n,1),3),
            "mean_words":round(sum(r["n_words"] for r in res)/max(n,1),1),
            "degenerate_rate":round(sum(1 for r in res if r["n_words"]<4)/max(n,1),3),
            "samples":res[:6]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", default="qwen2vl2b")
    ap.add_argument("--n", type=int, default=40)
    a=ap.parse_args(); random.seed(0)
    anns=[json.load(open(f)) for f in glob.glob(str(HC/"annotations/*.json"))]
    anns=[x for x in anns if x.get("hallu_adv") or x.get("hallu_pop")]
    random.shuffle(anns); anns=anns[:a.n]
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
    base=run(model,proc,anns); print("[base   ]",{k:base[k] for k in ("n","halluc_rate","mean_words","degenerate_rate")}, flush=True)
    del model; torch.cuda.empty_cache()
    model, proc = FastVisionModel.from_pretrained(a.adapter, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in a.adapter.lower():
        # transformers>=4.47 llava-hf: explicit vision config needed for token expansion
        try:
            proc.patch_size = 14
            proc.vision_feature_select_strategy = "default"
            if hasattr(proc, "num_additional_image_tokens"):
                proc.num_additional_image_tokens = 1
        except Exception:
            pass

    FastVisionModel.for_inference(model); model.eval()
    tuned=run(model,proc,anns); print("[tuned  ]",{k:tuned[k] for k in ("n","halluc_rate","mean_words","degenerate_rate")}, flush=True)
    (OUT/f"utility_{a.tag}.json").write_text(json.dumps({"base":base,"tuned":tuned},indent=2))
    print("[wrote]",OUT/f"utility_{a.tag}.json", flush=True)

if __name__=="__main__":
    main()
