# DUAL-Compass

**Direction-Aware Simultaneous Learning and Unlearning for Explainable VLM Hallucination Mitigation**

Full code, derived data, trained adapters, and per-run result files for DUAL-Compass — with a **table-by-table reproduction guide** so every number in the paper can be regenerated from this repository.

> Hallucination in vision–language models (VLMs) is usually treated as an *unsigned* scalar to minimize. DUAL-Compass treats it as a **signed** phenomenon — over-claiming (*yes-bias*) vs. over-withholding (*no-bias*) — and turns a direction-aware diagnosis of the model's *own* errors into the training signal: an **unlearn** stream (over-claims, suppressed with NPO), a **learn** stream (missed objects, reinforced with SFT), and a **retain** stream, optimized *simultaneously* under a **compass-guided closed loop** that re-diagnoses the signed score during training and stops at directional balance.

## ✅ Reproducibility at a glance

Every experiment in the paper is regenerable from the scripts here. This repo ships:
- **all training/eval/diagnosis scripts** (`code/`, 16 files),
- **all diagnosed streams and the direction-balanced expansion** (`data/streams/`, `data/expanded_queries.jsonl`),
- **the official POPE (COCO) question files** (`data/pope_official/`),
- **89 per-run result JSONs** that back every table (`data/results/`),
- **trained LoRA adapters** for every headline model (`adapters/`),
- deterministic seeds (seed 0 unless a multi-seed run is stated).

Only the *source images* are not redistributed (licenses); see [Data & images](#data--images).

---

## Reproducing every table / figure in the paper

Run everything from `code/`. `<M>` is a HF model id: `unsloth/Qwen2-VL-2B-Instruct`, `unsloth/Qwen2.5-VL-3B-Instruct`, or `unsloth/llava-1.5-7b-hf`. Every command writes a JSON under `dual/out/` whose committed copy is in `data/results/`.

### Figure 1 — Framework overview
Schematic (TikZ in the paper source); no data to reproduce.

### Table: RQ1 — direction-aware training (DUAL vs unlearn-only vs learn-only)
Claim: only the *simultaneous* objective works; learn-only collapses to 100% FP, unlearn-only is inert.
```bash
python train_dual_unsloth.py --mode dual         --seed 0 --steps 150 --eval_n 250   # → result_qwen2vl2b_dual.json
python train_dual_unsloth.py --mode unlearn_only --seed 0                            # → result_qwen2vl2b_unlearn_only.json
python train_dual_unsloth.py --mode learn_only   --seed 0                            # → result_qwen2vl2b_learn_only.json
# multi-seed (caption stats): repeat --mode {unlearn_only,learn_only,dual} --seed {1,2}
```
Result files: `data/results/result_qwen2vl2b_{dual,unlearn_only,learn_only}[_s1,_s2].json` (fields: `before`/`after` FP/FN/acc/signed).

### Table: loss-agnostic control (NPO vs GA, fixed vs loop)
Claim: under fixed weights both NPO and GA collapse when stressed; under the loop both are stable → the collapse is about feedback, not the loss.
```bash
# fixed-weight, stressed
python train_dual_unsloth.py --mode efuf_style --ga_w 0.3 --steps 450     # GA(0.3) 450 steps
python train_dual_unsloth.py --mode efuf_style --ga_w 1.0 --steps 150     # GA(1.0) collapse
python train_dual_unsloth.py --mode dual --lambda_u 3.0 --steps 150       # NPO(λu=3) collapse
# under the loop
python train_dual_loop.py --model <M> --tag qwen2vl2b_ext --unlearn_mode npo --gamma 0.3 --lr 3e-5 --probe_every 20
python train_dual_loop.py --model <M> --tag qwen2vl2b_ext --unlearn_mode ga  --gamma 0.3 --lr 3e-5 --probe_every 20
```
Result files: `data/results/result_qwen2vl2b_ext_efuf_style_*.json`, `..._loop_{npo,ga}_g0.3_lr3e-05.json`.

### Table: RQ2 — closed-loop control across models **+ multi-seed reliability (0/9 collapses)**
Claim: the damped loop with best-balance checkpointing succeeds on all three models and never collapses across 3 seeds each (9 runs).
```bash
for S in 0 1 2; do
  python train_dual_loop.py --model unsloth/Qwen2-VL-2B-Instruct  --tag qwen2vl2b_ext --gamma 0.3 --lr 3e-5 --probe_every 20 --unlearn_mode npo --seed $S
  python train_dual_loop.py --model unsloth/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b   --gamma 0.3 --lr 3e-5 --probe_every 20 --unlearn_mode npo --seed $S
  python train_dual_loop.py --model unsloth/llava-1.5-7b-hf        --tag llava7b      --gamma 0.3 --lr 3e-5 --probe_every 20 --unlearn_mode npo --seed $S
done
```
Result files: `data/results/result_{qwen2vl2b_ext,qwen25vl3b,llava7b}_loop_npo_g0.3_lr3e-05[_s1,_s2].json`.
Numbers: 2B 86.5±0.6, 3B 87.3±2.2, 7B 84.3±0.6 (mean±std over seeds 0/1/2); **0/9 collapses**.

### Table: EFUF-style vs DUAL-Compass
Claim: EFUF-style suppression-only matches peak accuracy but overshoots into no-bias and is budget-fragile.
```bash
python train_dual_unsloth.py --mode efuf_style --ga_w 0.3 --steps 150     # → result_qwen2vl2b_ext_efuf_style_s0.json
python eval_pope_compat.py --model <M> --adapter out/adapter_2b_efuf_s0 --tag 2b_efuf_s0   # → pope_2b_efuf_s0.json
```
Result files: `data/results/result_qwen2vl2b_ext_efuf_style_s0.json`, `pope_2b_efuf_s0.json`.

### Table: Human audit (3 annotators, 40 items, Fleiss κ≥0.6)
Human study using the questionnaire in the paper's Appendix "Human-Audit Questionnaire." Not a script — it is a blind human evaluation; the instrument (verbatim) and the aggregate numbers (direction-label agreement 90%, preference 85%, faithfulness 87/100, hallucination-free 89%) are in the paper. Caption/image sampling protocol: `code/eval_utility.py` (the 40-image utility subset).

### Table: small-sample direction inversion
Behavioral signed score on the released per-cell subsets vs. the published judge-based DSA. Computed from the underlying HalluCompass per-cell result files (analysis; the panel numbers are tabulated in the paper).

### Table: official POPE (COCO val2014) — external validation
Claim: direction is distribution-dependent; the suite-trained loop helps the yes-biased 7B and is direction-mismatched on the no-biased Qwen models.
```bash
# base vs suite-trained loop adapter, on official POPE
python eval_pope_official.py --model <M>                                   --tag <m>_base   # → popeofficial_<m>_base.json
python eval_pope_official.py --model <M> --adapter adapters/<m>_loop       --tag <m>_loop   # → popeofficial_<m>_loop.json
```
Result files: `data/results/popeofficial_{2b,3b,7b}_{base,loop}.json` (per-split + overall acc/F1/yes-ratio/FP/FN/signed).

### Table: **constructive POPE-COCO** (distribution-matched repair)
Claim: re-diagnosing the Qwen models *on COCO* (learn-heavy) and running the loop **improves** held-out POPE-COCO — 2B 73.0→91.3 (+18.4pp).
```bash
# 1) split 500 POPE-COCO images 300-train / 200-held-out; diagnose on the 300 → learn-heavy streams
python build_cocopope_streams.py --model <M> --tag cocopope2b            # → dual_full_cocopope2b.jsonl
# 2) train the loop on the COCO streams
python train_dual_loop.py --model <M> --tag cocopope2b --gamma 0.3 --lr 3e-5 --probe_every 20 --unlearn_mode npo --save_adapter out/adapter_cocopope2b
# 3) evaluate base vs COCO-loop on the DISJOINT held-out 200 images (greedy)
python eval_pope_official.py --model <M>                              --tag cocopope2b_base_ho --held_out --per_split 400
python eval_pope_official.py --model <M> --adapter out/adapter_cocopope2b --tag cocopope2b_loop_ho --held_out --per_split 400
```
Result files: `data/results/popeofficial_cocopope{2b,3b}_{base,loop}_ho.json`; streams `data/streams/streams_cocopope{2b,3b}.jsonl`; adapters `adapters/cocopope{2b,3b}_loop/`.
`--held_out` restricts evaluation to the last 200 of the 500 sorted images (disjoint from the 300 used for diagnosis/training) — no train/eval leakage.

### Table: utility audit + CHAIR (RQ3, generation quality preserved)
```bash
python eval_utility.py --model <M> --adapter <A> --tag <tag>    # absent-object mention rate, length, degeneration → utility_<tag>.json
python eval_chair.py   --model <M> [--adapter <A>] --tag <tag>  # CHAIR_s / CHAIR_i (n=200)               → chair_<tag>.json
```
Result files: `data/results/utility_*.json`, `chair_*.json`.

### Table: POPE-compatible balanced eval (greedy sanity check)
```bash
python eval_pope_compat.py --model <M> [--adapter <A>] --tag <tag>   # acc/P/R/F1/yes-ratio → pope_<tag>.json
```

### Table: DASH-B (honest negative — external transfer unsolved)
```bash
python eval_dashb.py --model <M> [--adapter <A>] --tag <tag>
```
Reported as a bounded negative result in the paper's appendix; scripts `build_dash_heavy.py` / `export_dashb_train.py` build the DASH-diagnosed streams.

---

## End-to-end pipeline (from scratch)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers peft unsloth pillow
# 1. direction-balanced data expansion (deterministic, seed 0)
python code/gen_extra_queries.py
# 2. diagnose the subject model -> unlearn/learn/retain streams
python code/expand_any.py --model unsloth/Qwen2-VL-2B-Instruct --tag qwen2vl2b_ext
# 3. train (fixed-weight ablations OR the closed loop) — see per-table commands above
# 4. evaluate — see per-table commands above
```
Tested with **PyTorch 2.10 (CUDA 12.8), Transformers 5.5.0, PEFT 0.19.1, Unsloth 2026.7.4**, single H100. Evaluation is sampling-free (log-likelihood) or greedy; training is deterministic under the stated seed.

## Repository layout

```
code/       16 scripts: diagnosis (expand_any), expansion (gen_extra_queries),
            training (train_dual_unsloth = fixed-weight; train_dual_loop = closed loop, has --seed),
            constructive COCO (build_cocopope_streams), and evaluation
            (eval_pope_official [--held_out], eval_pope_compat, eval_chair, eval_utility, eval_dashb)
data/
  streams/          diagnosed unlearn/learn/retain streams per model (incl. COCO-diagnosed)
  expanded_queries.jsonl   rule-based direction-balanced expansion
  pope_official/    official POPE question files (random/popular/adversarial, COCO val2014)
  results/          89 per-run metric JSONs backing every table
adapters/   trained LoRA adapters: {qwen2vl2b,qwen25vl3b,llava7b}_loop + cocopope{2b,3b}_loop
paper/      compiled paper PDF
```

## Data & images

Derived artifacts and adapters are included. **Source images are not redistributed** (licenses). The presence-probe pipeline expects the HalluCompass image release; official POPE / constructive-COCO expect COCO val2014 images (`http://images.cocodataset.org/val2014/`). Set the image path (`SNAP` / `IMG`) in the eval scripts to your local copies. Image licenses: MS-COCO val2014 (CC-BY 4.0), AMBER (research-only), NoCaps (CC-BY-SA), VizWiz (CC-BY 4.0, blind-user images, research terms).

## Licenses

- **Code**: MIT (`LICENSE`).
- **Derived data** (streams, queries, results): CC-BY 4.0 (`LICENSE-DATA`).
- Source images retain their original licenses and are not included.

## Citation

```bibtex
@inproceedings{dualcompass,
  title     = {DUAL-Compass: Direction-Aware Simultaneous Learning and Unlearning for Explainable VLM Hallucination Mitigation},
  booktitle = {Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '27)},
  year      = {2027}
}
```
