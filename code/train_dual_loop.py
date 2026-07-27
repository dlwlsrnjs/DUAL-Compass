"""Compass-guided CLOSED-LOOP DUAL training.
Every K steps: probe signed score s on a held-out probe set ->
  p_unlearn = clip(0.5 + 0.6*s, 0.15, 0.85)   (yes-bias -> more unlearning)
  stop early when |s| <= eps twice consecutively (converged to balance).
Saves adapter + history. Compare vs fixed-λ dual (no adaptation, no early stop).
"""
import os
os.environ.setdefault("USE_TF","0"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
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
        p = Path(SNAP)/sub/fname
        if p.exists(): return p
    h = list(Path(SNAP).rglob(fname)); return h[0] if h else None

class _null:
    def __enter__(self): return None
    def __exit__(self,*a): return False

def load(tag, splits):
    rows=[json.loads(l) for l in (OUT/f"dual_full_{tag}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["split"] in splits]

_IMG_CACHE={}
def _load_img(fname):
    if fname not in _IMG_CACHE:
        im=Image.open(find_image(fname)).convert("RGB"); im.thumbnail((512,512)); _IMG_CACHE[fname]=im
    return _IMG_CACHE[fname]

def ans_ce(model, proc, r, answer, use_ref=False):
    msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":r["query"]}]}]
    prompt=proc.apply_chat_template(msg, add_generation_prompt=True)
    image=_load_img(r["image"])
    enc=proc(images=[image], text=prompt+" "+answer, return_tensors="pt").to(model.device)
    encp=proc(images=[image], text=prompt, return_tensors="pt")
    plen=encp["input_ids"].shape[1]
    labels=enc["input_ids"].clone(); labels[:,:plen]=-100
    ctx=model.disable_adapter() if use_ref else _null()
    with ctx: out=model(**enc, labels=labels)
    n=(labels!=-100).sum().item()
    return -out.loss*max(n,1), out.loss

def signed_on(model, proc, rows):
    fp=fn=tp=tn=0
    model.eval()
    with torch.no_grad():
        for r in rows:
            ly,_=ans_ce(model,proc,r,"Yes."); ln,_=ans_ce(model,proc,r,"No.")
            pred="yes" if ly>ln else "no"
            if r["label"]=="no" and pred=="yes": fp+=1
            elif r["label"]=="yes" and pred=="no": fn+=1
            elif r["label"]=="yes": tp+=1
            else: tn+=1
    model.train()
    s=(fp-fn)/max(fp+fn,1)
    return s, dict(fp=fp,fn=fn,tp=tp,tn=tn,fp_rate=round(fp/max(fp+tn,1),3),
                   fn_rate=round(fn/max(fn+tp,1),3),signed=round(s,3),
                   acc=round((tp+tn)/max(fp+fn+tp+tn,1),3))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--tag", default="qwen2vl2b")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--probe_every", type=int, default=25)
    ap.add_argument("--probe_n", type=int, default=80)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--unlearn_mode", default="npo", choices=["npo","ga"])
    ap.add_argument("--ga_w", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_per_stream", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=250)
    ap.add_argument("--probe_tag", default="")
    ap.add_argument("--save_adapter", default="")
    ap.add_argument("--seed", type=int, default=0)
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

    model = FastVisionModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=False)

    random.seed(a.seed); torch.manual_seed(a.seed)  # re-seed: unsloth reseeds RNG during model/peft load
    tr=load(a.tag, ["rand","pop","extra"])
    random.shuffle(tr)
    if a.probe_tag:
        ptr=load(a.probe_tag, ["rand","pop","extra"]); random.shuffle(ptr)
    else:
        ptr=tr
    # balanced probe set held out from training
    pyes=[r for r in ptr if r["label"]=="yes"][:a.probe_n//2]
    pno =[r for r in ptr if r["label"]=="no"][:a.probe_n//2]
    probe=pyes+pno; probe_ids={id(r) for r in probe}
    pool_src=[r for r in tr if id(r) not in probe_ids]
    unl=[r for r in pool_src if r["stream"]=="unlearn"][:a.max_per_stream]
    lrn=[r for r in pool_src if r["stream"]=="learn"][:a.max_per_stream]
    ret=[r for r in pool_src if r["stream"]=="retain"][:a.max_per_stream]
    ev=load(a.tag,["adv"])[:a.eval_n]
    print(f"[data:{a.tag}] unlearn={len(unl)} learn={len(lrn)} retain={len(ret)} probe={len(probe)} eval={len(ev)}", flush=True)

    _,before=signed_on(model,proc,ev); print("[eval:before]",before, flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    FastVisionModel.for_training(model); model.train()

    p_unl=0.5; hist=[]; consec=0; step=0; best_ckpt=None
    while step<a.max_steps:
        if step % a.probe_every == 0:
            s,pd=signed_on(model,proc,probe)
            p_unl=max(0.15,min(0.85, 0.5+a.gamma*s))
            hist.append({"step":step,"probe_signed":round(s,3),"p_unlearn":round(p_unl,3),
                         "probe_acc":pd["acc"]})
            key = (abs(s), -pd["acc"])
            if best_ckpt is None or key < best_ckpt[0]:
                best_ckpt = (key, step, {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items() if "lora" in k.lower()})
            print(f"[probe] step={step} s={s:+.3f} -> p_unlearn={p_unl:.2f} acc={pd['acc']}", flush=True)
            consec = consec+1 if abs(s)<=a.eps else 0
            if consec>=2:
                print(f"[converged] |s|<={a.eps} twice at step {step}", flush=True)
                break
        u=random.random()
        if u < p_unl*2/3 and unl:      stream,r="unlearn",random.choice(unl)
        elif u < 2/3 and lrn:          stream,r="learn",random.choice(lrn)
        else:                          stream,r="retain",random.choice(ret)
        opt.zero_grad()
        if stream=="unlearn" and a.unlearn_mode=="ga":
            _,ce=ans_ce(model,proc,r,"Yes.")
            loss=-a.ga_w*ce
        elif stream=="unlearn":
            lp,_=ans_ce(model,proc,r,"Yes.")
            with torch.no_grad(): lpr,_=ans_ce(model,proc,r,"Yes.",use_ref=True)
            loss=-(2.0/a.beta)*F.logsigmoid(-a.beta*(lp-lpr))
        else:
            _,ce=ans_ce(model,proc,r,r["gold_answer"]); loss=ce
        loss.backward(); opt.step(); step+=1

    if best_ckpt is not None:
        # restore the most balanced probe checkpoint (smallest |s|, ties by probe acc)
        sd = model.state_dict()
        sd.update({k: v.to(next(model.parameters()).device) for k, v in best_ckpt[2].items()})
        model.load_state_dict(sd)
        print(f"[restore] best-balance checkpoint from step {best_ckpt[1]} (|s|={best_ckpt[0][0]:.3f}, probe_acc={-best_ckpt[0][1]:.3f})", flush=True)
    _,after=signed_on(model,proc,ev); print("[eval:after ]",after, flush=True)
    delta={k:round(after[k]-before[k],3) for k in ("fp_rate","fn_rate","acc","signed")}
    print("[DELTA]",delta, flush=True)
    res={"mode":"loop","model":a.model,"tag":a.tag,"steps_run":step,"history":hist,
         "before":before,"after":after,"delta":delta}
    _pbsuf = ("_pb"+a.probe_tag[-6:]) if a.probe_tag else ""
    _ssuf = f"_s{a.seed}" if a.seed != 0 else ""
    (OUT/f"result_{a.tag}_loop_{a.unlearn_mode}_g{a.gamma}_lr{a.lr}{_pbsuf}{_ssuf}.json").write_text(json.dumps(res,indent=2))
    if a.save_adapter:
        model.save_pretrained(a.save_adapter); proc.save_pretrained(a.save_adapter)
        print("[saved adapter]",a.save_adapter, flush=True)
    print("[wrote loop result]", flush=True)

if __name__=="__main__":
    main()
