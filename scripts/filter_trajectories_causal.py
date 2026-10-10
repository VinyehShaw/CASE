#!/usr/bin/env python3
"""Offline causal-linkage filter for vista3d trajectories.

Scores each trajectory's Phase 4 insight via causal_judge (evidence→insight→answer
causal chain). Keeps high-linkage trajectories regardless of answer correctness —
drops hallucinated reasoning (lucky-guess-correct AND bad-reasoning-wrong alike),
retains good-reasoning-wrong-pick (judge checks "insight supports correct answer
when student is wrong").

Pre-req: kill local vLLM so the judge uses dashscope qwen3.7-plus (not the SFT
model, which is biased toward its own training data):
    pkill -f "vllm serve"
    export OPENAI_API_KEY="sk-xxx"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

Usage:
    python scripts/filter_trajectories_causal.py \
        --trajectory-dir /tmp/mys/vista3d_traj_1000 \
        --threshold 0.4 \
        --max-samples 100        # omit for full run
        --output-manifest /tmp/mys/vista3d_traj_1000_kept.jsonl
"""
import os
import sys
import json
import glob
import argparse
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from distill.opd_rl.verl_opsd.causal_judge import evaluate_causal_evidence_batch
from distill.opd_rl.reward_insight import extract_phase


def extract_final_insight(messages):
    """Get Phase 4 insight text from the last non-tool-call assistant message."""
    final = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            final = m.get("content", "")
            if not isinstance(final, str):
                final = " ".join(
                    b.get("text", "") for b in final
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            break
    if not final:
        return ""
    return extract_phase(final, "Phase 4", "Phase 5")


def load_trajectories(trajectory_dir, max_samples=None):
    items = []
    skipped_no_tool = skipped_no_insight = 0
    for fp in sorted(glob.glob(os.path.join(trajectory_dir, "agent_*.jsonl"))):
        for line in open(fp):
            e = json.loads(line)
            msgs = e.get("messages", [])
            has_tool = any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)
            if not has_tool:
                skipped_no_tool += 1
                continue
            insight = extract_final_insight(msgs)
            if not insight or len(insight) < 20:
                skipped_no_insight += 1
                continue
            items.append({
                "patient_id": e.get("patient_id"),
                "question_idx": e.get("question_idx"),
                "question": e.get("question", ""),
                "options": e.get("options", {}),
                "ground_truth": e.get("ground_truth", ""),
                "student_answer": e.get("answer", "") or "",
                "student_insight": insight,
                "correct": e.get("correct"),
                "_file": os.path.basename(fp),
            })
            if max_samples and len(items) >= max_samples:
                return items, skipped_no_tool, skipped_no_insight
    return items, skipped_no_tool, skipped_no_insight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="subset size for quick estimate (omit = full run)")
    ap.add_argument("--output-manifest", type=str, default=None,
                    help="write JSONL of kept (patient_id, question_idx, score) pairs")
    ap.add_argument("--judge-model", type=str, default="qwen3-max",
                    help="LLM judge model (dashscope: qwen3-max / qwen3.7-plus)")
    args = ap.parse_args()

    items, sk_no_tool, sk_no_insight = load_trajectories(
        args.trajectory_dir, args.max_samples)
    print(f"Loaded {len(items)} trajectories with tool calls + insight "
          f"(skipped: {sk_no_tool} no-tool, {sk_no_insight} no-insight)")
    if not items:
        print("Nothing to score. Exiting.")
        return

    judge_items = [
        {
            "question": it["question"],
            "options": it["options"],
            "ground_truth": it["ground_truth"],
            "student_answer": it["student_answer"],
            "student_insight": it["student_insight"],
        }
        for it in items
    ]
    print(f"Scoring {len(judge_items)} via causal_judge (model={args.judge_model})...")
    scores = evaluate_causal_evidence_batch(
        judge_items, mode="api", judge_model=args.judge_model)

    kept = []
    for it, sc in zip(items, scores):
        it["causal_score"] = sc
        if sc >= args.threshold:
            kept.append(it)

    n = len(items)
    nk = len(kept)
    correct_total = sum(1 for it in items if it["correct"])
    wrong_total = n - correct_total
    correct_kept = sum(1 for it in kept if it["correct"])
    wrong_kept = nk - correct_kept
    scores_sorted = sorted(scores)

    print(f"\n{'='*60}")
    print(f"Causal-linkage filter results (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"total judged:      {n}")
    print(f"kept (>= thr):     {nk} ({nk/n*100:.1f}%)")
    print(f"dropped (< thr):   {n-nk} ({(n-nk)/n*100:.1f}%)")
    print(f"score: min={scores_sorted[0]:.2f} p25={scores_sorted[n//4]:.2f} "
          f"p50={scores_sorted[n//2]:.2f} p75={scores_sorted[int(n*0.75)]:.2f} "
          f"max={scores_sorted[-1]:.2f} mean={statistics.mean(scores):.2f}")
    print(f"\nbreakdown by answer correctness:")
    print(f"  correct-answer: {correct_total} total -> {correct_kept} kept "
          f"({correct_kept/max(correct_total,1)*100:.1f}%)")
    print(f"  wrong-answer:   {wrong_total} total -> {wrong_kept} kept "
          f"({wrong_kept/max(wrong_total,1)*100:.1f}%)")
    print(f"\n  >> wrong-answer-but-good-reasoning retained: {wrong_kept} "
          f"(these teach reasoning structure; RL refines the conclusion)")
    print(f"  >> lucky-guess-correct dropped: {correct_total - correct_kept} "
          f"(correct answer but hallucinated/weak insight)")

    if args.output_manifest:
        with open(args.output_manifest, "w") as f:
            for it in kept:
                f.write(json.dumps({
                    "patient_id": it["patient_id"],
                    "question_idx": it["question_idx"],
                    "causal_score": it["causal_score"],
                    "correct": it["correct"],
                    "file": it["_file"],
                }) + "\n")
        print(f"\nkept manifest written to {args.output_manifest}")


if __name__ == "__main__":
    main()
