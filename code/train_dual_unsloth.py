"""DUAL-Compass — direction-aware simultaneous learn+unlearn with unsloth (Qwen2-VL-2B).

Three-term DUAL loss on binary presence questions:
  L_learn (no-bias): CE toward correct 'Yes.' on missed present objects
  L_unlearn (yes-bias): NPO-suppress hallucinated 'Yes.' on absent objects (ref = adapter-disabled base)
  L_retain (faithful): CE on correct answers (utility anchor)
Ablations --mode: dual | unlearn_only | learn_only | sft_all
Eval: FP (over-claim), FN (missed), signed score, acc on held-out eval_split.
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("USE_FLAX","0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json, argparse, random
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F
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

class _null:
    def __enter__(self): return None
    def __exit__(self,*a): return False

def load(tag, splits):
    rows=[json.loads(l) for l in (OUT/f"dual_full_{tag}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["split"] in splits]

def ans_ce(model, proc, r, answer, use_ref=False):
    msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":r["query"]}]}]
    prompt=proc.apply_chat_template(msg, add_generation_prompt=True)
    image=Image.open(find_image(r["image"])).convert("RGB"); image.thumbnail((512,512))
    enc=proc(images=[image], text=prompt+" "+answer, return_tensors="pt").to(model.device)
    encp=proc(images=[image], text=prompt, return_tensors="pt")
    plen=encp["input_ids"].shape[1]
    labels=enc["input_ids"].clone(); labels[:,:plen]=-100
    ctx=model.disable_adapter() if use_ref else _null()
    with ctx:
        out=model(**enc, labels=labels)
    n=(labels!=-100).sum().item()
    return -out.loss*max(n,1), out.loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--tag", default="qwen2vl2b")
    ap.add_argument("--mode", default="dual", choices=["dual","unlearn_only","learn_only","sft_all","efuf_style"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_adapter", default="")
    ap.add_argument("--ga_w", type=float, default=0.3)
    ap.add_argument("--train_splits", nargs="+", default=["rand","pop"])
    ap.add_argument("--eval_split", default="adv")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_u", type=float, default=1.0)
    ap.add_argument("--lambda_r", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_per_stream", type=int, default=120)
    ap.add_argument("--eval_n", type=int, default=250)
    a=ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

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

    model = FastVisionModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.05,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=False)

    tr=load(a.tag, a.train_splits)
    unl=[r for r in tr if r["stream"]=="unlearn"][:a.max_per_stream]
    lrn=[r for r in tr if r["stream"]=="learn"][:a.max_per_stream]
    ret=[r for r in tr if r["stream"]=="retain"][:a.max_per_stream]
    print(f"[data] mode={a.mode} unlearn={len(unl)} learn={len(lrn)} retain={len(ret)} eval={a.eval_split}", flush=True)

    ev=load(a.tag,[a.eval_split])[:a.eval_n]
    def eval_dir():
        FastVisionModel.for_inference(model); model.eval()
        fp=fn=tp=tn=0
        with torch.no_grad():
            for r in ev:
                ly,_=ans_ce(model,proc,r,"Yes."); ln,_=ans_ce(model,proc,r,"No.")
                pred="yes" if ly>ln else "no"
                if r["label"]=="no" and pred=="yes": fp+=1
                elif r["label"]=="yes" and pred=="no": fn+=1
                elif r["label"]=="yes": tp+=1
                else: tn+=1
        FastVisionModel.for_training(model); model.train()
        s=(fp-fn)/max(fp+fn,1)
        return dict(fp=fp,fn=fn,tp=tp,tn=tn,fp_rate=round(fp/max(fp+tn,1),3),
                    fn_rate=round(fn/max(fn+tp,1),3),signed=round(s,3),
                    acc=round((tp+tn)/max(fp+fn+tp+tn,1),3))

    before=eval_dir(); print("[eval:before]",before, flush=True)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    pool=[]
    if a.mode in ("dual","unlearn_only","sft_all","efuf_style"): pool+=[("unlearn",r) for r in unl]
    if a.mode in ("dual","learn_only","sft_all"):   pool+=[("learn",r) for r in lrn]
    if a.mode in ("dual","sft_all","efuf_style"):   pool+=[("retain",r) for r in ret]
    random.seed(a.seed); torch.manual_seed(a.seed)  # re-seed: unsloth reseeds RNG during model load
    random.shuffle(pool)
    FastVisionModel.for_training(model); model.train()
    step=0
    while step<a.steps and pool:
        stream,r=pool[step%len(pool)]; opt.zero_grad()
        if stream=="unlearn" and a.mode=="sft_all":
            # balanced SFT baseline: train gold "No." on over-claimed items (no NPO)
            stream="sft_gold"
            _,ce=ans_ce(model,proc,r,r["gold_answer"])
            loss=ce
        elif stream=="unlearn" and a.mode=="efuf_style":
            # EFUF-adapted-to-binary: plain gradient ascent on the hallucinated
            # assertion with EFUF's 0.3 negative-loss weight; retain = sentence loss.
            _,ce=ans_ce(model,proc,r,"Yes.")
            loss=-a.ga_w*ce
        elif stream=="unlearn":
            lp,_=ans_ce(model,proc,r,"Yes.")
            with torch.no_grad():
                lpr,_=ans_ce(model,proc,r,"Yes.",use_ref=True)
            loss=a.lambda_u*(-(2.0/a.beta)*F.logsigmoid(-a.beta*(lp-lpr)))
        elif stream=="sft_gold":
            pass
        else:
            # for sft_all, unlearn-stream records reach here with gold answer "No." (balanced SFT)
            _,ce=ans_ce(model,proc,r,r["gold_answer"])
            loss=(a.lambda_r if stream=="retain" else 1.0)*ce
        loss.backward(); opt.step()
        if step%30==0: print(f"  step {step:3d} [{stream:7s}] loss={loss.item():.3f}", flush=True)
        step+=1

    after=eval_dir(); print("[eval:after ]",after, flush=True)
    delta={"fp_rate":round(after["fp_rate"]-before["fp_rate"],3),
           "fn_rate":round(after["fn_rate"]-before["fn_rate"],3),
           "acc":round(after["acc"]-before["acc"],3),
           "signed":round(after["signed"]-before["signed"],3)}
    print("[DELTA]", delta, flush=True)
    res={"mode":a.mode,"model":a.model,"train_splits":a.train_splits,"eval_split":a.eval_split,
         "steps":a.steps,"before":before,"after":after,"delta":delta}
    (OUT/f"result_{a.tag}_{a.mode}_s{a.seed}_st{a.steps}_g{a.ga_w}.json" if (a.mode=="efuf_style" and (a.steps!=150 or a.ga_w!=0.3)) or a.steps!=150 else OUT/f"result_{a.tag}_{a.mode}_s{a.seed}.json").write_text(json.dumps(res,indent=2))
    if a.save_adapter:
        model.save_pretrained(a.save_adapter); proc.save_pretrained(a.save_adapter)
        print("[saved adapter]",a.save_adapter, flush=True)
    print("[wrote]",OUT/f"result_{a.tag}_{a.mode}_s{a.seed}.json", flush=True)

if __name__=="__main__":
    main()
