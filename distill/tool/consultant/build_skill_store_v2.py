#!/usr/bin/env python3
"""Aggregate harvested medboard skills into a retrieval store (v2 — industry-standard convergence).

Evolution of build_skill_store.py's merge_harvest_by_organ:
  v1 weakness (measured on s1912 cold-start harvest, 4748 raw skills):
    - organ-only buckets are too coarse (general=2632, heart=914 mix
      hypertension/MI/valve topics)
    - guidance "frequency vote" is a no-op on free-form text: every guidance
      is unique, _most_common degenerates to alphabetical pick (top vote = 1)
  v2 design:
    1. two-level buckets: organ × topic (classified from patient_type+keywords)
    2. within-bucket dedup by keyword-Jaccard >= 0.5; keep the longest
       guidance as the representative (most complete methodology statement)
       → measured: heart×hypertension 789→11, general×respiratory 451→17
    3. leak filter: drop skills whose guidance contains a hard answer-flip
       instruction ("answer 'Yes'/'No'", "select 'Yes'") — failure reflections
       are written AFTER seeing GT, so such lines encode the test-set answer
       distribution and systematically flip Y/N questions in the next round
       (measured: Q6 -26.1pts, Q7 -23.5pts when unfiltered)
    4. per-bucket cap (default 3): match_skills returns top-3, so each bucket
       contributes at most what retrieval can surface

Usage:
    python build_skill_store_v2.py \
        --harvest-dir <eval_output>/medboard \
        --output <dir>/skill_store_merged.jsonl \
        [--seed ...] [--jaccard 0.5] [--per-bucket-cap 3] [--stats]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

LEAK_PAT = re.compile(
    r"(?:answer|select|choose|reply|respond)\s+(?:with\s+)?['\"]?\b(yes|no)\b", re.I)

TOPIC_RULES = [
    ("hypertension", ("hypertension", "i10", "blood pressure")),
    ("diabetes", ("diabet",)),
    ("cancer", ("cancer", "malignan", "tumor", "tumour", "oncolog", "metasta")),
    ("respiratory", ("respiratory", "asthma", "allerg", "hayfever", "eczema",
                     "emphysema", "bronch", "copd", "dvt", "pulmonary")),
    ("cardio", ("cardio", "myocardial", "infarct", "valve", "cardiomyo",
                "arrhythmia", "atrial")),
    ("icd", ("icd", "z-code", "diagnosis code", "disease system")),
    ("steatosis", ("steato", "fatty liver", "liver fat")),
    ("atropy_volume", ("atrophy", "volumetry", "volume change")),
]


def _norm_organ(o):
    return (o or "").strip().lower() or "general"


def _topic_of(sk):
    t = (sk.get("patient_type", "") + " " +
         " ".join(sk.get("keywords") or [])).lower()
    for name, keys in TOPIC_RULES:
        if any(k in t for k in keys):
            return name
    return "other"


def _sig(sk):
    return frozenset((k or "").strip().lower() for k in (sk.get("keywords") or [])[:6])


def _is_leaky(sk):
    return bool(LEAK_PAT.search(sk.get("guidance", "") or ""))


def load_harvest(harvest_dir):
    records = []
    for fp in sorted(glob.glob(os.path.join(harvest_dir, "skills_*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.extend(rec.get("skills", []))
    return records


def build(records, jaccard_thr=0.5, per_bucket_cap=3):
    # stage 1: filter empty + leak
    n_leak = 0
    clean = []
    for sk in records:
        if not (sk.get("guidance") or "").strip():
            continue
        if _is_leaky(sk):
            n_leak += 1
            continue
        clean.append(sk)

    # stage 2: organ × topic buckets
    buckets = defaultdict(list)
    for sk in clean:
        buckets[(_norm_organ(sk.get("organ")), _topic_of(sk))].append(sk)

    # stage 3: within-bucket jaccard dedup, longest guidance kept as representative
    store = []
    n_before_dedup = 0
    for (organ, topic) in sorted(buckets):
        group = buckets[(organ, topic)]
        n_before_dedup += len(group)
        kept = []
        for sk in sorted(group, key=lambda s: -len(s.get("guidance", ""))):
            s = _sig(sk)
            if any(len(s & _sig(k)) / max(1, len(s | _sig(k))) >= jaccard_thr
                   for k in kept):
                continue
            kept.append(sk)
        # cap: retrieval surfaces top-3, a bucket shouldn't exceed that
        kept = kept[:per_bucket_cap]
        for i, sk in enumerate(kept):
            store.append({
                "skill_id": f"mb_v2_{organ}_{topic}_{i+1}",
                "organ": organ,
                "keywords": [k for k in (sk.get("keywords") or [])[:8] if k],
                "patient_type": (sk.get("patient_type") or "")[:120],
                "guidance": (sk.get("guidance") or "")[:600],
                "pitfalls": (sk.get("pitfalls") or "")[:400],
                "source": sk.get("source", ""),
                "bucket": f"{organ}x{topic}",
                "bucket_size": len(group),
            })
    stats = {
        "raw": len(records),
        "leaky_filtered": n_leak,
        "after_filter": len(clean),
        "buckets": len(buckets),
        "store_size": len(store),
    }
    return store, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True,
                    help="dir with skills_*.jsonl (an eval output medboard dir)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", default="",
                    help="optional seed store jsonl to prepend")
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--per-bucket-cap", type=int, default=3)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    records = load_harvest(args.harvest_dir)
    if not records:
        print(f"[skill_v2] no harvested skills in {args.harvest_dir}")
        sys.exit(1)

    store, stats = build(records, args.jaccard, args.per_bucket_cap)

    out = []
    if args.seed and os.path.exists(args.seed) and os.path.getsize(args.seed) > 0:
        with open(args.seed) as f:
            content = f.read().strip()
        if content:
            out.extend(json.loads(content) if content.startswith("[")
                       else [json.loads(l) for l in content.splitlines() if l.strip()])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for sk in out + store:
            f.write(json.dumps(sk, ensure_ascii=False) + "\n")

    print(f"[skill_v2] raw={stats['raw']} leak_filtered={stats['leaky_filtered']} "
          f"buckets={stats['buckets']} -> store={len(out)+len(store)} "
          f"-> {args.output}")
    if args.stats:
        from collections import Counter
        c = Counter(s["bucket"] for s in store)
        for b, n in c.most_common(10):
            print(f"   {b:<24} {n}")
        print(f"   source: {dict(Counter(s['source'] for s in store))}")


if __name__ == "__main__":
    main()
