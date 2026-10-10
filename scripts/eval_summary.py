#!/usr/bin/env python3
"""Per-module + per-question-type accuracy summary for eval dense_*.json dirs.

Usage:
    python scripts/eval_summary.py <label1>:<dense_dir1> [<label2>:<dense_dir2> ...]
"""
import json, glob, os, sys
from collections import defaultdict


def qtype(qtxt, nopt, gt):
    t = (qtxt or "").lower()
    if "cancer" in t:
        return "Q1 cancer(Y/N)"
    if "diabetes" in t:
        return "Q2 diabetes(Y/N)"
    if "cardiovascular" in t:
        return "Q3-multi cardio" if nopt == 5 else "Q3-single cardio"
    if "respiratory or allergic" in t:
        return "Q4-multi resp" if nopt == 6 else "Q4-single resp"
    if "disease systems" in t:
        return "Q5-multi ICD-system"
    if "icd-10" in t and ("personal history" in t or "family history" in t):
        return "Q6 ICD-history(Y/N)"
    if "hypertension" in t or "essential" in t:
        return "Q7 hypertension(Y/N)"
    if "icd-10" in t:
        return "Q5-single ICD-cond"
    return "other"


ORDER = [
    "Q1 cancer(Y/N)", "Q2 diabetes(Y/N)",
    "Q3-multi cardio", "Q3-single cardio",
    "Q4-multi resp", "Q4-single resp",
    "Q5-multi ICD-system", "Q5-single ICD-cond",
    "Q6 ICD-history(Y/N)", "Q7 hypertension(Y/N)",
    "other",
]


def norm(s):
    return ",".join(sorted(x.strip().upper() for x in str(s).replace(" ", ",").split(",") if x.strip()))


def main():
    runs = [a.split(":", 1) for a in sys.argv[1:]]
    stats = {}
    for label, ddir in runs:
        ts = defaultdict(lambda: [0, 0])
        n = t = 0
        n_pat = 0
        for f in glob.glob(os.path.join(ddir, "dense_*.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            n_pat += 1
            for q in d.get("questions", []):
                gt = q.get("ground_truth") or q.get("solution")
                if not gt:
                    continue
                t += 1
                ok = norm(q.get("answer", "")) == norm(gt)
                n += ok
                qt = qtype(q.get("question", ""), len(q.get("options", {}) or {}), gt)
                ts[qt][1] += 1
                ts[qt][0] += ok
        stats[label] = (n, t, ts, n_pat)
        acc = 100 * n / t if t else 0
        print(f"  {label:24s}: {n_pat} patients, {n}/{t} = {acc:.1f}%")

    accs = [100 * stats[l][0] / stats[l][1] for l, _ in runs if stats[l][1]]
    if accs:
        print(f"  {'MEAN (arith of agents)':24s}: {sum(accs) / len(accs):.1f}%  "
              f"(unweighted over {len(accs)} agents; pooled COMBINED below)")

    mods = [m for m, _ in runs]
    live = [m for m in mods if any(stats[m][2].get(q, [0, 0])[1] for q in ORDER)]
    if not live:
        return
    print("\nPer-Question-Type Accuracy (acc%, correct/total per module)")
    print(f"  {'type':22s}" + "".join(f"{m[:9]:>16s}" for m in mods))
    for qt in ORDER:
        if not any(stats[m][2].get(qt, [0, 0])[1] for m in mods):
            continue
        row = f"  {qt:22s}"
        for m in mods:
            c, t = stats[m][2].get(qt, [0, 0])
            cell = f"{100 * c / t:.1f}% {c}/{t}" if t else "-"
            row += f"{cell:>16s}"
        print(row)
    all_c = sum(stats[m][0] for m in mods)
    all_t = sum(stats[m][1] for m in mods)
    if all_t:
        cell = f"{100 * all_c / all_t:.1f}% {all_c}/{all_t}"
        print(f"  {'COMBINED':22s}" + "".join(f"{'':>16s}" for _ in mods[:-1]) +
              f"{cell:>16s}")


if __name__ == "__main__":
    main()
