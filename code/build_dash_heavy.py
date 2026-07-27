"""DASH-heavy stream: suite + DASH records with DASH over-claim/negatives oversampled,
so the loop's balanced probe (drawn from training streams) actually contains DASH negatives.
"""
import json, argparse
from pathlib import Path
OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite_tag", default="qwen2vl2b_ext")
    ap.add_argument("--mix_tag", default="qwen2vl2b_mix")
    ap.add_argument("--out_tag", default="qwen2vl2b_dashheavy")
    ap.add_argument("--oversample", type=int, default=4)
    a=ap.parse_args()
    mix=[json.loads(l) for l in (OUT/f"dual_full_{a.mix_tag}.jsonl").read_text().splitlines() if l.strip()]
    suite=[r for r in mix if r.get("source")!="dashb"]
    dash=[r for r in mix if r.get("source")=="dashb"]
    dash_u=[r for r in dash if r["stream"]=="unlearn"]
    dash_neg=[r for r in dash if r["stream"] in ("unlearn","retain") and r["label"]=="no"]
    # oversample dash unlearn (the over-claim signal) and dash negatives (for probe TNR visibility)
    heavy = suite + dash*1 + dash_u*(a.oversample-1)
    (OUT/f"dual_full_{a.out_tag}.jsonl").write_text("\n".join(json.dumps(r) for r in heavy)+"\n")
    from collections import Counter
    c=Counter((r.get("source","suite")=="dashb", r["stream"]) for r in heavy)
    print(f"[dash-heavy] total={len(heavy)} suite={len(suite)} dash_base={len(dash)} dash_u_extra={len(dash_u)*(a.oversample-1)}")
    print("  composition:", {f"{'dash' if k[0] else 'suite'}_{k[1]}":v for k,v in c.items()})
if __name__=="__main__": main()
