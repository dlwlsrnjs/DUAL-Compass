"""DUAL-Compass Pilot B — direction-aware simultaneous learn+unlearn (model-agnostic).

Three-term DUAL loss on binary presence questions:
    L = L_learn (no-bias: push toward correct answer on missed present objects)
      + lambda_u * L_unlearn (yes-bias: NPO-suppress hallucinated 'Yes' on absent objects)
      + lambda_r * L_retain (faithful: CE on correct answers, preserve utility)
Reference for NPO = same LoRA model with adapter disabled (frozen base).
Ablations via --mode: dual | unlearn_only | learn_only | sft_all
"""
from __future__ import annotations
import os, json, argparse, random
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("USE_FLAX","0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

HC = Path("/home/ubuntu/342/jinkwon/orthocampus/HalluCompass")
DUAL_OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")

class _null:
    def __enter__(self): return None
    def __exit__(self,*a): return False

def find_image(fname, img_root):
    for sub in ["coco","nocaps","vizwiz","amber",""]:
        p = Path(img_root)/sub/fname
        if p.exists(): return p
    hits = list(Path(img_root).rglob(fname))
    return hits[0] if hits else None

def load_dual(model_safe, splits):
    rows=[json.loads(l) for l in (DUAL_OUT/f"dual_{model_safe}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["split"] in splits]

def ans_ce(model, processor, img, query, answer, img_root, device, use_ref=False):
    """Return (summed_logprob, mean_CE) for answer tokens given image+prompt."""
    conv=[{"role":"user","content":[{"type":"image"},{"type":"text","text":query+" Answer yes or no."}]}]
    prompt=processor.apply_chat_template(conv, add_generation_prompt=True)
    ip=find_image(img,img_root)
    image=Image.open(ip).convert("RGB"); image.thumbnail((384,384))
    enc=processor(images=[image], text=prompt+" "+answer, return_tensors="pt").to(device)
    encp=processor(images=[image], text=prompt, return_tensors="pt")
    plen=encp["input_ids"].shape[1]
    labels=enc["input_ids"].clone(); labels[:,:plen]=-100
    ctx=model.disable_adapter() if use_ref else _null()
    with ctx:
        out=model(**enc, labels=labels)
    n=(labels!=-100).sum().item()
    return -out.loss*max(n,1), out.loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--model_safe", default="HuggingFaceTB__SmolVLM2_2_2B_Instruct")
    ap.add_argument("--mode", default="dual", choices=["dual","unlearn_only","learn_only","sft_all"])
    ap.add_argument("--train_splits", nargs="+", default=["rand","pop"])
    ap.add_argument("--eval_split", default="adv")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_u", type=float, default=1.0)
    ap.add_argument("--lambda_r", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--max_per_stream", type=int, default=40)
    a=ap.parse_args()
    random.seed(0); torch.manual_seed(0); device="cuda"

    print(f"[load] {a.model_id}", flush=True)
    processor=AutoProcessor.from_pretrained(a.model_id, do_image_splitting=False)
    model=AutoModelForImageTextToText.from_pretrained(a.model_id, torch_dtype=torch.bfloat16).to(device)
    lcfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model=get_peft_model(model, lcfg); model.print_trainable_parameters()

    tr=load_dual(a.model_safe, a.train_splits)
    unl=[r for r in tr if r["stream"]=="unlearn"][:a.max_per_stream]
    lrn=[r for r in tr if r["stream"]=="learn"][:a.max_per_stream]
    ret=[r for r in tr if r["stream"]=="retain"][:a.max_per_stream]
    print(f"[data] mode={a.mode} unlearn={len(unl)} learn={len(lrn)} retain={len(ret)} eval={a.eval_split}", flush=True)

    def eval_dir(split):
        rows=load_dual(a.model_safe,[split]); fp=fn=tp=tn=0; model.eval()
        with torch.no_grad():
            for r in rows:
                ly,_=ans_ce(model,processor,r["image"],r["query"],"Yes.",a.img_root,device)
                ln,_=ans_ce(model,processor,r["image"],r["query"],"No.",a.img_root,device)
                pred="yes" if ly>ln else "no"
                if r["label"]=="no" and pred=="yes": fp+=1
                elif r["label"]=="yes" and pred=="no": fn+=1
                elif r["label"]=="yes": tp+=1
                else: tn+=1
        s=(fp-fn)/max(fp+fn,1)
        return dict(fp=fp,fn=fn,tp=tp,tn=tn,fp_rate=round(fp/max(fp+tn,1),3),
                    fn_rate=round(fn/max(fn+tp,1),3),signed=round(s,3),
                    acc=round((tp+tn)/max(fp+fn+tp+tn,1),3))

    before=eval_dir(a.eval_split); print("[eval:before]",before, flush=True)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    pool=[]
    if a.mode in ("dual","unlearn_only","sft_all"): pool+=[("unlearn",r) for r in unl]
    if a.mode in ("dual","learn_only","sft_all"):   pool+=[("learn",r) for r in lrn]
    if a.mode in ("dual","sft_all"):                pool+=[("retain",r) for r in ret]
    random.shuffle(pool)
    model.train(); step=0
    while step<a.steps and pool:
        stream,r=pool[step%len(pool)]; opt.zero_grad()
        if stream=="unlearn":
            lp,_=ans_ce(model,processor,r["image"],r["query"],"Yes.",a.img_root,device)
            with torch.no_grad():
                lpr,_=ans_ce(model,processor,r["image"],r["query"],"Yes.",a.img_root,device,use_ref=True)
            loss=a.lambda_u*(-(2.0/a.beta)*F.logsigmoid(-a.beta*(lp-lpr)))
        else:
            gold="Yes." if r["label"]=="yes" else "No."
            _,ce=ans_ce(model,processor,r["image"],r["query"],gold,a.img_root,device)
            loss=(a.lambda_r if stream=="retain" else 1.0)*ce
        loss.backward(); opt.step()
        if step%20==0: print(f"  step {step:3d} [{stream:7s}] loss={loss.item():.3f}", flush=True)
        step+=1
    after=eval_dir(a.eval_split); print("[eval:after ]",after, flush=True)
    print("[DELTA]", {"fp_rate":round(after["fp_rate"]-before["fp_rate"],3),
                      "fn_rate":round(after["fn_rate"]-before["fn_rate"],3),
                      "acc":round(after["acc"]-before["acc"],3),
                      "signed":round(after["signed"]-before["signed"],3)}, flush=True)
    res={"mode":a.mode,"model":a.model_safe,"before":before,"after":after}
    outp=DUAL_OUT/f"result_{a.model_safe}_{a.mode}.json"; outp.write_text(json.dumps(res,indent=2))
    print("[wrote]",outp, flush=True)

if __name__=="__main__":
    main()
