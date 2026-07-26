# DUAL-Compass

**Direction-Aware Simultaneous Learning and Unlearning for Explainable VLM Hallucination Mitigation**

Code, derived data, and trained adapters for the DUAL-Compass framework.

> Hallucination in vision–language models (VLMs) is usually treated as an *unsigned* scalar to be minimized. DUAL-Compass instead treats it as a **signed** phenomenon — over-claiming (*yes-bias*) vs. over-withholding (*no-bias*) — and elevates a direction-aware diagnosis of a model's *own* errors into the training signal. The model's errors are partitioned into an **unlearn** stream (over-claims, suppressed with NPO), a **learn** stream (missed objects, reinforced with SFT), and a **retain** stream (faithful anchors), optimized *simultaneously* under one objective. A **compass-guided closed loop** re-diagnoses the signed score during training and re-balances the two directional pressures until the error direction converges to zero, returning the most balanced checkpoint visited.

## Key findings

- **Simultaneity is required.** Each single-direction pressure alone is inert (unlearn-only) or collapses the model (learn-only → 100% false-positive).
- **The collapse is about feedback, not the loss.** Under fixed weights both NPO and gradient-ascent unlearning collapse when pushed; under closed-loop control both are stable. The diagnosis-driven controller is the contribution.
- **Direction is distribution-dependent.** On official POPE (COCO) the models' diagnosed direction matches our source-direction gradient; repair transfers exactly when direction does.
- Validated across three VLMs (Qwen2-VL-2B, Qwen2.5-VL-3B, LLaVA-1.5-7B), a standard external benchmark (POPE/COCO), CHAIR utility, and a blind human audit.

## Repository layout

```
code/       training, expansion, and evaluation scripts
data/
  streams/          diagnosis->stream files (unlearn/learn/retain) per model
  expanded_queries.jsonl   rule-based direction-balanced data expansion
  pope_official/    official POPE question files (random/popular/adversarial)
  results/          per-run metric JSONs backing the paper's tables
adapters/   trained LoRA adapters (closed-loop, per model)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers peft unsloth pillow
```
Tested with PyTorch 2.10 (CUDA 12.8), Transformers 5.5.0, PEFT 0.19.1, Unsloth 2026.7.4, on a single H100.

## Reproduction

**1. Direction-balanced data expansion** (rule-based, deterministic under seed 0):
```bash
python code/gen_extra_queries.py
```

**2. Diagnosis -> streams** (one forward pass per query; produces `data/streams/*.jsonl`):
```bash
python code/expand_any.py --model unsloth/Qwen2-VL-2B-Instruct --tag qwen2vl2b
```

**3a. Fixed-weight ablations** (DUAL / unlearn-only / learn-only / balanced-SFT / EFUF-style):
```bash
python code/train_dual_unsloth.py --mode {dual|unlearn_only|learn_only|sft_all|efuf_style} \
    --seed 0 --steps 150 --eval_n 250
```

**3b. Compass-guided closed loop** (damped controller with best-balance checkpointing):
```bash
python code/train_dual_loop.py --model unsloth/Qwen2-VL-2B-Instruct --tag qwen2vl2b \
    --gamma 0.3 --lr 3e-5 --probe_every 20
```

**4. Evaluation**
```bash
python code/eval_pope_compat.py    --model <m> [--adapter adapters/<a>]   # POPE-compatible balanced set
python code/eval_pope_official.py  --model <m> [--adapter adapters/<a>]   # official POPE on COCO val2014
python code/eval_chair.py          --model <m> [--adapter adapters/<a>]   # CHAIR caption hallucination
python code/eval_utility.py        --model <m> --adapter adapters/<a>     # free-form utility audit
```

## Data & images

This release contains **derived artifacts only** (diagnosed streams, expanded queries, results) and **trained adapters**. It does **not** redistribute source images. The presence-probe pipeline expects the underlying HalluCompass image release; the official POPE evaluation expects COCO val2014 images (`http://images.cocodataset.org/val2014/`). Point the image paths in the eval scripts to your local copies.

## Licenses

- **Code**: MIT (see `LICENSE`).
- **Derived data** (streams, queries, results): CC-BY 4.0 (see `LICENSE-DATA`).
- Source images retain their original licenses (MS-COCO, AMBER, NoCaps, VizWiz) and are **not** included here.

## Citation

```bibtex
@inproceedings{dualcompass,
  title     = {DUAL-Compass: Direction-Aware Simultaneous Learning and Unlearning for Explainable VLM Hallucination Mitigation},
  booktitle = {Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '27)},
  year      = {2027}
}
```
