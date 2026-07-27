"""DUAL-Compass Pilot A — build direction-aware learn/unlearn/retain training data."""
from __future__ import annotations
import json, glob
from pathlib import Path

HC = Path("/home/ubuntu/342/jinkwon/orthocampus/HalluCompass")
RESULTS = HC / "results"
ANN = HC / "annotations"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

PRETTY = {
    "HuggingFaceTB__SmolVLM_Instruct": "SmolVLM-Instruct-2.25B",
    "HuggingFaceTB__SmolVLM2_2_2B_Instruct": "SmolVLM2-2.2B",
    "Qwen__Qwen2_VL_2B_Instruct": "Qwen2-VL-2B",
    "Qwen__Qwen2_5_VL_3B_Instruct": "Qwen2.5-VL-3B",
    "llava_hf__llava_1.5_7b_hf": "LLaVA-1.5-7B",
    "llava_hf__llava_onevision_qwen2_7b_ov_hf": "LLaVA-OV-7B",
    "HuggingFaceM4__Idefics3_8B_Llama3": "Idefics3-8B",
    "OpenGVLab__InternVL2_2B": "InternVL2-2B",
    "microsoft__Phi_3_5_vision_instruct": "Phi-3.5-vision",
    "THUDM__glm_4v_9b": "GLM-4V-9B",
}

def load_annotations():
    idx = {}
    for f in glob.glob(str(ANN / "*.json")):
        a = json.loads(Path(f).read_text()); idx[a["filename"]] = a
    return idx

def build_for_model(model_safe, ann):
    records, fp, fn, tp, tn = [], 0, 0, 0, 0
    for cell in sorted(RESULTS.glob(f"{model_safe}__id_POPE_*__plain.jsonl")):
        split = cell.stem.split("id_POPE_")[1].split("__")[0]
        for line in cell.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line); label, pred = r["label"], r["pred"]
            a = ann.get(r["image"], {})
            obj = (r["text"].replace("Is there a ", "").replace("Is there an ", "")
                   .replace(" in the image?", "").strip())
            base = {"image": r["image"], "object": obj, "query": r["text"],
                    "split": split, "label": label, "model_pred": pred,
                    "model_raw": r.get("raw", ""), "scene_summary": a.get("scene_summary", "")}
            if label == "no" and pred == "yes":
                fp += 1; records.append({**base, "stream": "unlearn", "direction": "yes_bias",
                    "gold_answer": "No.", "note": "over-claimed absent object"})
            elif label == "yes" and pred == "no":
                fn += 1; records.append({**base, "stream": "learn", "direction": "no_bias",
                    "gold_answer": "Yes.", "note": "missed present object"})
            elif label == pred:
                if label == "yes": tp += 1
                else: tn += 1
                records.append({**base, "stream": "retain", "direction": "faithful",
                    "gold_answer": "Yes." if label == "yes" else "No.", "note": "correct anchor"})
    s_beh = (fp - fn) / max(fp + fn, 1)
    stats = {"model": PRETTY.get(model_safe, model_safe), "n_total": fp+fn+tp+tn,
             "n_unlearn": fp, "n_learn": fn, "n_retain": tp+tn,
             "fp_rate": round(fp/max(fp+tn,1),4), "fn_rate": round(fn/max(fn+tp,1),4),
             "signed_score_behavioral": round(s_beh,4),
             "dominant_direction": "yes_bias" if s_beh>0 else ("no_bias" if s_beh<0 else "balanced")}
    return records, stats

def main():
    ann = load_annotations()
    models = sorted({p.stem.split("__id_POPE")[0] for p in RESULTS.glob("*__id_POPE_*__plain.jsonl")})
    dsa = {}
    dp = HC / "outputs" / "dsa_routing.json"
    if dp.exists():
        for k, v in json.loads(dp.read_text()).items():
            if isinstance(v, dict) and "signed_score" in v: dsa[k] = v["signed_score"]
    all_stats = []
    for m in models:
        recs, st = build_for_model(m, ann)
        (OUT / f"dual_{m}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        st["dsa_signed_paper"] = None; pn = PRETTY.get(m, "")
        for k, v in dsa.items():
            if pn and (pn.lower() in str(k).lower() or pn.replace("-","").lower() in str(k).replace("-","").lower()):
                st["dsa_signed_paper"] = round(v, 3)
        all_stats.append(st)
    (OUT / "dual_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"{'model':<22}{'N':>6}{'unlearn':>9}{'learn':>7}{'retain':>8}{'s_beh':>8}{'dir':>10}{'DSA':>8}")
    print("-"*80)
    for st in all_stats:
        print(f"{st['model']:<22}{st['n_total']:>6}{st['n_unlearn']:>9}{st['n_learn']:>7}"
              f"{st['n_retain']:>8}{st['signed_score_behavioral']:>8}{st['dominant_direction']:>10}"
              f"{str(st['dsa_signed_paper']):>8}")
    tu=sum(s['n_unlearn'] for s in all_stats); tl=sum(s['n_learn'] for s in all_stats); tr=sum(s['n_retain'] for s in all_stats)
    print("-"*80); print(f"{'TOTAL':<22}{tu+tl+tr:>6}{tu:>9}{tl:>7}{tr:>8}")
    print(f"\nwrote {OUT}/dual_<model>.jsonl and dual_stats.json")

if __name__ == "__main__":
    main()
