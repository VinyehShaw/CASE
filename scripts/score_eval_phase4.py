#!/usr/bin/env python3
"""Offline Phase-4 causal-judge scoring of eval outputs (dashscope, free kimi-k2.6).

Runs the SAME judge used in RL training (causal_judge.APICausalJudge -> dashscope
chat.completions) over every question of an eval run, then aggregates:
  - per-agent mean judge score (subscore by agent)
  - per-question-type mean judge score (Q1-Q7 subscores)
  - overall pooled mean (overall evaluation score)
  - judge score | answer-correct vs answer-wrong (sanity: good reasoning tracks correctness)

Phase-4 "insight" is read per agent from the eval dense_*.json:
  medboard   : q["phases"]["reflective_insight"]
  vista3d    : q["trajectory"]["phases"][phase_name==insight_synthesis]["consensus_insight"]
  medpanel_v2: q["trajectory"]["consensus_insight"]

Usage:
  python scripts/score_eval_phase4.py --eval-dir <OUTPUT_BASE> \
      [--judge-model kimi-k2.6] [--batch-size 4] [--max-workers 64] \
      [--max-samples N] [--out <file>]
"""
import argparse
import glob
import json
import os
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from eval_summary import qtype, ORDER  # noqa: E402  (same Q1-Q7 taxonomy as the acc summary)
from distill.opd_rl.verl_opsd import causal_judge  # noqa: E402

MODULES = ["vista3d", "medpanel_v2", "medboard"]


def extract_insight(mod, q):
    """Pull the Phase-4 insight text for a question, per agent's dense layout."""
    if mod == "medboard":
        return ((q.get("phases") or {}).get("reflective_insight") or "").strip()
    tr = q.get("trajectory") or {}
    if isinstance(tr, dict):
        ci = (tr.get("consensus_insight") or "").strip()
        if ci:
            return ci
        ph = tr.get("phases")
        if isinstance(ph, list):
            for p in ph:
                if isinstance(p, dict) and p.get("phase_name") == "insight_synthesis":
                    return (p.get("consensus_insight") or "").strip()
        if isinstance(ph, dict):
            for k in ("insight_synthesis", "reflective_insight", "consensus_insight"):
                if ph.get(k):
                    return str(ph[k]).strip()
    return ""


def load_module(eval_dir, mod, max_samples=None):
    d = os.path.join(eval_dir, mod)
    items, skipped = [], 0
    for f in sorted(glob.glob(os.path.join(d, "dense_*.json"))):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for q in data.get("questions", []):
            gt = q.get("ground_truth") or q.get("solution")
            if not gt:
                continue
            ins = extract_insight(mod, q)
            qt = qtype(q.get("question", ""), len(q.get("options", {}) or {}), gt)
            rec = {
                "question": q.get("question", ""),
                "options": q.get("options", {}) or {},
                "ground_truth": str(gt),
                "student_answer": str(q.get("answer", "") or ""),
                "student_insight": ins,
                "_qtype": qt, "_correct": bool(q.get("correct")), "_mod": mod,
            }
            if len(ins) < 20:
                skipped += 1
                continue
            items.append(rec)
            if max_samples and len(items) >= max_samples:
                return items, skipped
    return items, skipped


def preflight_local_judge():
    """causal_judge hijacks localhost:8000/8001 if a vLLM judge is up. Warn."""
    for port in (8001, 8000):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"[WARN] local vLLM judge detected on :{port} — judge will use it, "
                  f"NOT dashscope. Stop it to force dashscope kimi.")
            return
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="OUTPUT_BASE containing vista3d/medpanel_v2/medboard")
    ap.add_argument("--judge-model", default="kimi-k2.6")
    ap.add_argument("--judge-path", choices=["single", "batch"], default="single",
                    help="single = score_single/CAUSAL_JUDGE_PROMPT strict 0-0.9 (RL active path); "
                         "batch = score_batch_api lenient 0-1 (fewer calls, higher scores)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None, help="per-module cap (pilot)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "")
    os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not args.out:
        args.out = os.path.join(args.eval_dir, "phase4_judge_summary.txt")

    preflight_local_judge()
    judge = causal_judge.get_causal_judge(
        mode="api", judge_model=args.judge_model, judge_max_concurrent=args.max_workers)
    print(f"[judge] {type(judge).__name__} model={getattr(judge,'model','?')} base_url={getattr(judge,'base_url','?')}")

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit(f"Phase-4 Causal Judge Scoring  (model={args.judge_model}, path={args.judge_path}, dashscope)")
    emit(f"Eval dir: {args.eval_dir}")
    emit("=" * 60)

    per_mod = {}
    qtype_scores = defaultdict(lambda: defaultdict(list))  # qt -> mod -> [scores]
    all_scores, all_correct = [], []
    for mod in MODULES:
        items, skipped = load_module(args.eval_dir, mod, args.max_samples)
        if not items:
            emit(f"  {mod:12s}: 0 scoreable questions (skipped {skipped} no/short insight)")
            per_mod[mod] = (0, skipped, 0.0)
            continue
        judge_items = [{k: it[k] for k in
                        ("question", "options", "ground_truth", "student_answer", "student_insight")}
                       for it in items]
        if args.judge_path == "batch":
            results = judge.score_batch_api(judge_items, batch_size=args.batch_size,
                                            max_workers=args.max_workers)
            n_fb = sum(1 for r in results if not r or "raw" not in r)
        else:
            results = judge.score_batch(judge_items)  # concurrent score_single == RL active path
            n_fb = sum(1 for r in results if not r or "error" in (r or {}))
        scores = [r.get("score", 0.5) if r else 0.5 for r in results]
        mean = sum(scores) / len(scores)
        per_mod[mod] = (len(scores), skipped, mean)
        for it, s in zip(items, scores):
            qtype_scores[it["_qtype"]][mod].append(s)
            all_scores.append(s); all_correct.append(it["_correct"])
        emit(f"  {mod:12s}: scored {len(scores)} (skipped {skipped} no-insight, "
             f"{n_fb} parse-fallback) | mean judge score = {mean:.3f}")

    # ── means ──
    agent_means = [m for (_, _, m) in per_mod.values() if m]
    scored_mods = [mod for mod in MODULES if per_mod.get(mod, (0,))[0]]
    emit("-" * 60)
    if agent_means:
        emit(f"  MEAN (arith of agents) : {sum(agent_means)/len(agent_means):.3f}")
    if all_scores:
        emit(f"  OVERALL (pooled, all {len(all_scores)} scored) : {sum(all_scores)/len(all_scores):.3f}")

    # ── correctness split ──
    c = [s for s, ok in zip(all_scores, all_correct) if ok]
    w = [s for s, ok in zip(all_scores, all_correct) if not ok]
    if c and w:
        emit(f"  judge|correct-answer : {sum(c)/len(c):.3f} (n={len(c)}) | "
             f"judge|wrong-answer : {sum(w)/len(w):.3f} (n={len(w)})")

    # ── per-question-type table ──
    emit("")
    emit("Per-Question-Type mean judge score")
    emit(f"  {'type':22s}" + "".join(f"{m.split('_')[0][:9]:>12s}" for m in scored_mods) + f"{'ALL':>12s}")
    for qt in ORDER:
        if not any(qtype_scores[qt].get(m) for m in scored_mods):
            continue
        row = f"  {qt:22s}"
        pool = []
        for m in scored_mods:
            v = qtype_scores[qt].get(m, [])
            row += f"{(sum(v)/len(v)):>12.3f}" if v else f"{'-':>12s}"
            pool += v
        row += f"{(sum(pool)/len(pool)):>12.3f}" if pool else f"{'-':>12s}"
        emit(row)
    emit(f"  {'n=':22s}" + "".join(f"{len([1 for qt in ORDER for s in qtype_scores[qt].get(m,[])]) :>12d}"
                                   for m in scored_mods))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
