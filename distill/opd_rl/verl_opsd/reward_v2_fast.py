"""
reward_v2_fast.py — Fast reward without LLM API calls.

Uses keyword overlap + structure matching for insight similarity.
Falls back to LLM judge if API key is available.

R_total = R_correct + R_format + alpha * R_insight_match + beta * R_causal
"""

import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from distill.opd_rl.reward_insight import (
    correctness_reward,
    format_reward,
    extract_phase,
)
from distill.opd_rl.reward import extract_answer


# Medical keywords for insight quality scoring
MEDICAL_KEYWORDS = [
    "lesion", "nodule", "mass", "tumor", "atrophy", "hypertrophy",
    "inflammation", "edema", "enhancement", "signal", "intensity",
    "enlargement", "narrowing", "occlusion", "patency", "thickening",
    "stable", "interval", "change", "progression", "regression",
    "comparison", "baseline", "follow-up", "visit",
    "findings", "clinical", "implication", "confidence",
    "radiological", "imaging", "mri", "modality",
]

INSIGHT_SECTIONS = ["Key Findings", "Temporal Changes", "Clinical Implications", "Confidence"]


def keyword_overlap_score(teacher_text, student_text):
    """Compute keyword overlap between teacher and student insight text."""
    if not teacher_text or not student_text:
        return 0.0
    
    teacher_lower = teacher_text.lower()
    student_lower = student_text.lower()
    
    # Keyword overlap
    teacher_kw = set(kw for kw in MEDICAL_KEYWORDS if kw in teacher_lower)
    student_kw = set(kw for kw in MEDICAL_KEYWORDS if kw in student_lower)
    
    if not teacher_kw:
        return 0.5
    
    overlap = len(teacher_kw & student_kw) / len(teacher_kw)
    return overlap


def structure_score(text):
    """Check if insight has the 4-section structure."""
    if not text:
        return 0.0
    found = sum(1 for section in INSIGHT_SECTIONS if section.lower() in text.lower())
    return found / len(INSIGHT_SECTIONS)


def length_ratio_score(teacher_text, student_text):
    """Score based on length ratio (student should produce similar-length insight)."""
    if not teacher_text or not student_text:
        return 0.0
    ratio = len(student_text) / max(len(teacher_text), 1)
    if ratio < 0.1:
        return 0.1
    elif ratio > 3.0:
        return 0.7
    elif 0.5 <= ratio <= 2.0:
        return 1.0
    else:
        return 0.5


def fast_insight_match(teacher_insight, student_insight):
    """Fast insight matching without API calls.
    
    Combines: keyword overlap (50%) + structure (30%) + length ratio (20%)
    """
    kw_score = keyword_overlap_score(teacher_insight, student_insight)
    struct_score = structure_score(student_insight)
    len_score = length_ratio_score(teacher_insight, student_insight)
    
    total = 0.5 * kw_score + 0.3 * struct_score + 0.2 * len_score
    return min(1.0, max(0.0, total))


# ─── Optional LLM judge (if API key available) ───

_llm_scorer = None

def _get_llm_scorer(**kwargs):
    global _llm_scorer
    if _llm_scorer is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("dummy"):
            return None  # Skip LLM if no valid key
        try:
            from distill.opd_rl.reward_insight import InsightRewardScorer
            _llm_scorer = InsightRewardScorer(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", ""),
                model=kwargs.get("insight_model", "qwen3.7-plus"),
                max_concurrent=kwargs.get("insight_max_concurrent", 20),
                cache_path=kwargs.get("insight_cache_path", ""),
            )
            print(f"[reward_v2_fast] LLM scorer initialized")
        except Exception:
            return None
    return _llm_scorer


# ─── Main reward function ───

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Reward with fast insight matching + causal bonus."""
    extra_info = extra_info or {}
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)

    alpha = kwargs.get("insight_alpha", 0.5)
    beta = kwargs.get("causal_beta", 0.5)
    use_llm = kwargs.get("use_llm_judge", False)

    # 1. Correctness
    r_correct = correctness_reward(solution_str, str(ground_truth).strip())
    is_correct = (r_correct == 1.0)

    # 2. Format
    r_format = format_reward(solution_str)

    # 3. Insight matching
    r_insight = 0.0
    teacher_insight = extra_info.get("teacher_insight", "")
    if alpha > 0 and teacher_insight:
        student_insight = extract_phase(solution_str, "Phase 4", "Phase 5")
        if student_insight and len(student_insight) > 20:
            # Try LLM judge first (if enabled and API key available)
            if use_llm:
                scorer = _get_llm_scorer(**kwargs)
                if scorer:
                    question = extra_info.get("question", "")
                    r_insight = scorer.score_single(
                        teacher_insight, student_insight, question, str(ground_truth)
                    )
                else:
                    r_insight = fast_insight_match(teacher_insight, student_insight)
            else:
                r_insight = fast_insight_match(teacher_insight, student_insight)

    # 4. Causal alignment
    r_causal = 0.0
    if is_correct and r_insight > 0.6:
        r_causal = 0.5
    elif not is_correct and r_insight  < 0.3 and r_insight > 0:
        r_causal = -0.5

    total = r_correct + r_format + alpha * r_insight + beta * r_causal
    return total
