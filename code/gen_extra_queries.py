"""Generate EXTRA binary presence queries from HalluCompass annotations.
absent  (label=no):  objects in hallu_pop + hallu_adv  (never in truth)
present (label=yes): objects in truth
Dedup against the released id_{rand,pop,adv} queries. Balanced sample capped by --cap.
"""
import json, glob, random, argparse
from pathlib import Path
HC = Path("/home/ubuntu/342/jinkwon/orthocampus/HalluCompass")
OUT = Path("/home/ubuntu/342/jinkwon/orthocampus/dual/out")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=3000)
    a = ap.parse_args(); random.seed(0)
    existing = set()
    for s in ["rand","pop","adv"]:
        for l in (HC/"queries"/f"id_{s}.jsonl").read_text().splitlines():
            if l.strip():
                d = json.loads(l); existing.add((d["filename"], d["object"].lower()))
    absent, present = [], []
    for f in glob.glob(str(HC/"annotations/*.json")):
        an = json.load(open(f)); fn = an["filename"]
        truth = {t.lower() for t in an.get("truth", [])}
        for o in an.get("hallu_pop", []) + an.get("hallu_adv", []):
            if o.lower() in truth or (fn, o.lower()) in existing: continue
            absent.append((fn, o, "no"))
        for o in an.get("truth", []):
            if (fn, o.lower()) in existing: continue
            present.append((fn, o, "yes"))
    random.shuffle(absent); random.shuffle(present)
    n = min(a.cap // 2, len(absent), len(present))
    rows = []
    for fn, o, lab in absent[:n] + present[:n]:
        art = "an" if o[0].lower() in "aeiou" else "a"
        rows.append({"filename": fn, "object": o, "label": lab, "split": "extra",
                     "query": f'Is there {art} {o} in this image? Answer with only "yes" or "no".'})
    random.shuffle(rows)
    (OUT/"extra_queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[extra] wrote {len(rows)} ({n} absent + {n} present), dedup vs {len(existing)} existing")

if __name__ == "__main__":
    main()
