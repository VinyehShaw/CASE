#!/usr/bin/env python3
"""
reward_v5_anatomist.py — Reward with LLM semantic causal judge (Vista3D variant).

Copy of reward_v5.py with judge_model passthrough for configurable judge model.
Supports both NaiveRewardManager (single sample) and BatchRewardManager (parallel).

R_total = R_correct + format_weight * R_format + beta * R_causal_judge + linkage
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from distill.opd_rl.reward_insight import (
    correctness_reward,
    format_reward,
    extract_phase,
)
from distill.opd_rl.reward import extract_answer
from distill.opd_rl.verl_opsd.causal_judge import evaluate_causal_evidence, evaluate_causal_evidence_batch


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    """Reward with LLM semantic causal judge.

    Supports both reward managers:
    - NaiveRewardManager: data_source, solution_str, ground_truth, extra_info (singular)
    - BatchRewardManager: data_sources, solution_strs, ground_truths, extra_infos (plural, via kwargs)
    """
    if "solution_strs" in kwargs:
        return _compute_batch(
            data_sources=kwargs.pop("data_sources"),
            solution_strs=kwargs.pop("solution_strs"),
            ground_truths=kwargs.pop("ground_truths"),
            extra_infos=kwargs.pop("extra_infos"),
            **kwargs,
        )
    return _compute_single(data_source, solution_str, ground_truth, extra_info, **kwargs)


def _compute_single(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Single-sample reward (NaiveRewardManager path)."""
    extra_info = extra_info or {}
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)

    alpha = kwargs.get("insight_alpha", 0.5)
    beta = kwargs.get("causal_judge_beta", 0.5)
    judge_mode = kwargs.get("judge_mode", "auto")
    judge_model = kwargs.get("judge_model", "qwen3.7-plus")
    format_weight = kwargs.get("format_weight", 0.4)
    insight_correct_bonus = kwargs.get("insight_correct_bonus", 0.5)
    insight_wrong_penalty = kwargs.get("insight_wrong_penalty", 0.3)
    insight_threshold = kwargs.get("insight_threshold", 0.6)

    r_correct = correctness_reward(solution_str, str(ground_truth).strip())
    is_correct = (r_correct == 1.0)
    r_format = format_reward(solution_str, require_phase6=(data_source == "medboard_opsd"))

    r_insight = 0.0
    teacher_insight = extra_info.get("teacher_insight", "")
    if alpha > 0 and teacher_insight:
        from distill.opd_rl.verl_opsd.reward_v2_fast import fast_insight_match
        student_insight = extract_phase(solution_str, "Phase 4", "Phase 5")
        if student_insight and len(student_insight) > 20:
            r_insight = fast_insight_match(teacher_insight, student_insight)

    r_causal_judge = 0.0
    if beta > 0:
        if "precomputed_judge_score" in extra_info:
            r_causal_judge = extra_info["precomputed_judge_score"]
        else:
            student_insight = extract_phase(solution_str, "Phase 4", "Phase 5")
            if student_insight and len(student_insight) > 20:
                student_answer = extract_answer(solution_str) or ""
                question = extra_info.get("question", "")
                options = extra_info.get("options", {})
                try:
                    r_causal_judge = evaluate_causal_evidence(
                        question=question,
                        options=options,
                        ground_truth=str(ground_truth).strip(),
                        student_answer=student_answer,
                        student_insight=student_insight,
                        mode=judge_mode,
                        judge_model=judge_model,
                        judge_max_concurrent=kwargs.get("judge_max_concurrent", 150),
                    )
                except Exception as e:
                    print(f"[reward_v5_anatomist] causal_judge error: {e}, using 0.5")
                    r_causal_judge = 0.5

    total = r_correct + format_weight * r_format + beta * r_causal_judge

    insight_good = r_causal_judge > insight_threshold
    linkage = 0.0
    if insight_good and is_correct:
        linkage = insight_correct_bonus
        total += linkage
    elif insight_good and not is_correct:
        linkage = -insight_wrong_penalty
        total += linkage

    return {
        "score": total,
        "r_correct": r_correct,
        "r_format": r_format,
        "r_judge": r_causal_judge,
        "linkage": linkage,
    }


def _compute_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Batch reward (BatchRewardManager path) — parallel judge calls."""
    n = len(solution_strs)

    alpha = kwargs.get("insight_alpha", 0.5)
    beta = kwargs.get("causal_judge_beta", 0.5)
    judge_mode = kwargs.get("judge_mode", "auto")
    judge_model = kwargs.get("judge_model", "qwen3.7-plus")
    format_weight = kwargs.get("format_weight", 0.4)
    insight_correct_bonus = kwargs.get("insight_correct_bonus", 0.5)
    insight_wrong_penalty = kwargs.get("insight_wrong_penalty", 0.3)
    insight_threshold = kwargs.get("insight_threshold", 0.6)
    insight_topk = int(kwargs.get("insight_topk", 2))
    insight_min_range = float(kwargs.get("insight_min_range", 0.05))

    r_corrects = []
    r_formats = []
    r_insights = []
    is_corrects = []
    parsed_eis = []

    judge_items = []
    judge_indices = []

    for i in range(n):
        sol = solution_strs[i]
        gt = str(ground_truths[i]).strip() if ground_truths and i < len(ground_truths) and ground_truths[i] else ""
        ei = extra_infos[i] if extra_infos and i < len(extra_infos) and extra_infos[i] else {}
        if isinstance(ei, str):
            ei = json.loads(ei)
        parsed_eis.append(ei)

        r_correct = correctness_reward(sol, gt)
        is_correct = (r_correct == 1.0)
        ds = data_sources[i] if data_sources and i < len(data_sources) else ""
        r_format = format_reward(sol, require_phase6=(ds == "medboard_opsd"))

        r_insight = 0.0
        teacher_insight = ei.get("teacher_insight", "")
        if alpha > 0 and teacher_insight:
            from distill.opd_rl.verl_opsd.reward_v2_fast import fast_insight_match
            si = extract_phase(sol, "Phase 4", "Phase 5")
            if si and len(si) > 20:
                r_insight = fast_insight_match(teacher_insight, si)

        if beta > 0:
            si = extract_phase(sol, "Phase 4", "Phase 5")
            if si and len(si) > 20:
                judge_items.append({
                    "question": ei.get("question", ""),
                    "options": ei.get("options", {}),
                    "ground_truth": gt,
                    "student_answer": extract_answer(sol) or "",
                    "student_insight": si,
                })
                judge_indices.append(i)

        r_corrects.append(r_correct)
        r_formats.append(r_format)
        r_insights.append(r_insight)
        is_corrects.append(is_correct)

    # Batch judge call (parallel via ThreadPoolExecutor inside score_batch_api)
    r_judges = [0.0] * n
    judge_failed_all = False
    if judge_items:
        t0 = time.time()
        try:
            judge_scores = evaluate_causal_evidence_batch(
                judge_items,
                mode=judge_mode,
                judge_model=judge_model,
                judge_max_concurrent=kwargs.get("judge_max_concurrent", 150),
                judge_batch_size=kwargs.get("judge_batch_size", 4),
                judge_max_workers=kwargs.get("judge_max_workers", 48),
            )
            for idx, score in zip(judge_indices, judge_scores):
                r_judges[idx] = score
        except Exception as e:
            judge_failed_all = True
            print(f"[reward_v5_anatomist] batch judge error: {e}, using 0.5", flush=True)
            for idx in judge_indices:
                r_judges[idx] = 0.5
        elapsed = time.time() - t0
        print(f"[JUDGE_BATCH] {len(judge_items)}/{n} judged in {elapsed:.1f}s "
              f"({n - len(judge_items)} skipped, no Phase 4)", flush=True)

    # Decide insight_good per sample. Primary: within-group (same question =
    # the GRPO rollouts of one prompt) top-k by judge score — self-adapts as
    # insight quality improves, unlike a fixed absolute threshold. Guards:
    # groups need >=2 judged members and a judge-score spread >= min_range;
    # unjudged samples (no Phase 4) never qualify; if the whole judge call
    # failed, fall back to the absolute threshold. insight_topk=0 disables.
    judge_index_set = set(judge_indices)
    insight_goods = [False] * n
    if insight_topk > 0 and not judge_failed_all:
        groups = {}
        for i in range(n):
            ei = parsed_eis[i]
            key = (str(ground_truths[i]).strip(), (ei.get("question") or "")[:200])
            groups.setdefault(key, []).append(i)
        for members in groups.values():
            judged = [i for i in members if i in judge_index_set]
            if len(judged) < 2:
                continue
            scores = [r_judges[i] for i in judged]
            if max(scores) - min(scores) < insight_min_range:
                continue
            top = sorted(judged, key=lambda i: r_judges[i], reverse=True)[:insight_topk]
            for i in top:
                insight_goods[i] = True
    else:
        insight_goods = [r_judges[i] > insight_threshold for i in range(n)]

    results = []
    for i in range(n):
        total = r_corrects[i] + format_weight * r_formats[i] + beta * r_judges[i]

        linkage = 0.0
        if insight_goods[i] and is_corrects[i]:
            linkage = insight_correct_bonus
            total += linkage
        elif insight_goods[i] and not is_corrects[i]:
            linkage = -insight_wrong_penalty
            total += linkage

        results.append({
            "score": total,
            "r_correct": r_corrects[i],
            "r_format": r_formats[i],
            "r_judge": r_judges[i],
            "linkage": linkage,
        })

    return results
