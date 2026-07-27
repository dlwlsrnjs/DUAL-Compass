"""Full held-out adversarial evaluation (ALL adv-split records, logprob probe), base or adapter."""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, argparse
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel

OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")
SNAP = "/home/ubuntu/.cache/huggingface/hub/datasets--anonymous80934--HalluCompass/snapshots/da2a24e1b7f0363a638942a84d825d3209bb49b9/images"

def find_image(fname):
    from pathlib import Path as _P
    if _P(fname).is_absolute() and _P(fname).exists(): return _P(fname)
    for sub in ["coco","nocaps","vizwiz","amber",""]:
        p=Path(SNAP)/sub/fname
        if p.exists(): return p
    h=list(Path(SNAP).rglob(fname)); return h[0] if h else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--stream_tag", default="qwen2vl2b_ext")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--skeptical", action="store_true")
    a=ap.parse_args()
    rows=[json.loads(l) for l in (OUT/f"dual_full_{a.stream_tag}.jsonl").read_text().splitlines() if l.strip()]
    rows=[r for r in rows if r["split"]=="adv"]
    print(f"[adv-full:{a.tag}] {len(rows)} queries, max_side={a.max_side}", flush=True)
    path=a.adapter if a.adapter else a.model
    model, proc = FastVisionModel.from_pretrained(path, load_in_4bit=False, dtype=torch.bfloat16)
    if "llava" in (a.adapter if a.adapter else a.model).lower():
        try: proc.patch_size=14; proc.vision_feature_select_strategy="default"
        except Exception: pass
    FastVisionModel.for_inference(model); model.eval()
    fp=fn=tp=tn=0
    with torch.no_grad():
        for i,r in enumerate(rows):
            qtext=r["query"]
            if a.skeptical:
                qtext="Be conservative. Only answer yes if you can clearly see the object; if unsure, answer no. "+qtext
            msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":qtext}]}]
            prompt=proc.apply_chat_template(msg, add_generation_prompt=True)
            im=Image.open(find_image(r["image"])).convert("RGB")
            if a.max_side>0: im.thumbnail((a.max_side,a.max_side))
            def lp(ans):
                enc=proc(images=[im], text=prompt+" "+ans, return_tensors="pt").to(model.device)
                encp=proc(images=[im], text=prompt, return_tensors="pt")
                plen=encp["input_ids"].shape[1]
                labels=enc["input_ids"].clone(); labels[:,:plen]=-100
                out=model(**enc, labels=labels)
                n=(labels!=-100).sum().item()
                return (-out.loss*max(n,1)).item()
            pred="yes" if lp("Yes.")>lp("No.") else "no"
            if r["label"]=="no" and pred=="yes": fp+=1
            elif r["label"]=="yes" and pred=="no": fn+=1
            elif r["label"]=="yes": tp+=1
            else: tn+=1
            if (i+1)%300==0: print(f"  {i+1}/{len(rows)}", flush=True)
    s=(fp-fn)/max(fp+fn,1); n=fp+fn+tp+tn
    res={"tag":a.tag,"n":n,"fp_rate":round(fp/max(fp+tn,1),4),"fn_rate":round(fn/max(fn+tp,1),4),
         "acc":round((tp+tn)/max(n,1),4),"signed":round(s,4),"max_side":a.max_side}
    print("[adv-full]",res, flush=True)
    (OUT/f"advfull_{a.tag}.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
