#!/usr/bin/env python3
"""
reward_insight.py — Insight relevance reward via LLM judge API.

Evaluates how relevant the student's Phase 4 insight is to answering the MCQ.
Score 0.0-1.0, cached per (question, insight) pair.

API: DashScope (Bailian platform)
  export OPENAI_API_KEY="sk-xxx"
  export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from openai import OpenAI

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from distill.opd_rl.reward import extract_answer


def extract_phase(response, phase_start_marker, phase_end_marker):
    """Extract text between two phase markers."""
    start = response.find(phase_start_marker)
    if start < 0:
        return ""
    end = response.find(phase_end_marker, start)
    if end < 0:
        end = len(response)
    return response[start:end].strip()


def _normalize_answer(answer):
    if not answer:
        return ""
    letters = sorted(set(l.strip().upper() for l in str(answer).split(",") if l.strip()))
    return ",".join(letters)


def correctness_reward(response, ground_truth):
    """R_correct: +1 correct, -1 wrong, -2 no answer."""
    answer = extract_answer(response)
    if answer is None:
        return -2.0
    elif _normalize_answer(answer) == _normalize_answer(ground_truth):
        return 1.0
    else:
        return -1.0


def format_reward(response, require_phase6=False):
    """R_format: +1.0 if all 5 phases present.

    require_phase6=True (medboard) additionally requires Phase 6 — without it,
    RL gradually drops the reflection/write_skill stage since nothing scores it.
    """
    phases = sum(1 for p in range(1, 6) if f"Phase {p}" in response)
    if phases != 5:
        return 0.0
    if require_phase6 and "Phase 6" not in response:
        return 0.0
    return 1.0


class RewardCache:
    """File-based cache for insight scores."""

    def __init__(self, cache_path=""):
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["score"]

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, score):
        self.cache[key] = score
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "score": score}) + "\n")

    @staticmethod
    def make_key(question, insight):
        content = question + insight
        return hashlib.md5(content.encode()).hexdigest()


class InsightRewardScorer:
    """Score insight relevance to MCQ via LLM judge API.

    API keys (Bailian platform DashScope):
      export OPENAI_API_KEY="sk-xxx"
      export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    """

    JUDGE_PROMPT = """You are evaluating a medical imaging insight for its relevance to answering a clinical question.

Question: {question}

Options:
{options}

Insight (Phase 4 Moderator Synthesis):
{insight}

Evaluate: How relevant and useful is this insight for arriving at the correct answer to the question?
Consider:
1. Does the insight address the key clinical question?
2. Does it identify the relevant findings that distinguish between options?
3. Does it lead logically toward the correct answer?

Output ONLY a single number between 0.0 and 1.0 (0=irrelevant, 1=highly relevant and decision-changing)."""

    def __init__(self, api_key="", base_url="", model="qwen3.7-plus",
                 max_concurrent=20, cache_path=""):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.model = model
        self.semaphore = Semaphore(max_concurrent)
        self.cache = RewardCache(cache_path)

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set:\n"
                '  export OPENAI_API_KEY="sk-xxx"  # Bailian platform API Key\n'
                '  export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"'
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
            timeout=300.0,
            max_retries=3,
        )

    def score_single(self, question, options, insight_text):
        """Score a single insight. Returns 0.0-1.0."""
        if not insight_text or len(insight_text) < 20:
            return 0.0

        key = RewardCache.make_key(question, insight_text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)
        prompt = self.JUDGE_PROMPT.format(
            question=question[:500],
            options=options_text[:500],
            insight=insight_text[:2000],
        )

        for attempt in range(5):
            with self.semaphore:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=10,
                    )
                    score_text = response.choices[0].message.content.strip()
                    score = float(score_text)
                    score = max(0.0, min(1.0, score))
                    self.cache.put(key, score)
                    return score
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        time.sleep(2 ** (attempt + 1))
                    elif attempt < 4:
                        time.sleep(2)
                    else:
                        return 0.5
        return 0.5

    def score_batch(self, items):
        """Score multiple insights in parallel.

        items: list of (question, options, insight_text)
        Returns: list of scores (0.0-1.0)
        """
        results = [0.5] * len(items)

        with ThreadPoolExecutor(max_workers=self.semaphore._value) as executor:
            futures = {
                executor.submit(self.score_single, q, o, i): idx
                for idx, (q, o, i) in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = 0.5

        return results


def compute_reward(response, ground_truth, question, options,
                   insight_scorer=None, alpha=0.5,
                   correct_r=1.0, wrong_r=-1.0, invalid_r=-2.0, format_r=0.2):
    """Compute total reward for a single response.

    R_total = R_correct + alpha * R_insight + R_format
    """
    r_correct = correctness_reward(response, ground_truth)
    r_format = format_reward(response)

    r_insight = 0.0
    if insight_scorer and alpha > 0:
        insight = extract_phase(response, "Phase 4", "Phase 5")
        if insight:
            r_insight = insight_scorer.score_single(question, options, insight)

    return r_correct + alpha * r_insight + r_format


def compute_reward_batch(responses, ground_truths, questions, options_list,
                         insight_scorer=None, alpha=0.5,
                         correct_r=1.0, wrong_r=-1.0, invalid_r=-2.0, format_r=0.2):
    """Compute rewards for a batch of responses.

    Correctness and format are computed locally (fast).
    Insight scores are computed in parallel via API (if insight_scorer provided).
    """
    # Fast: correctness + format
    rewards = []
    insight_items = []
    insight_indices = []

    for i, (resp, gt) in enumerate(zip(responses, ground_truths)):
        r_correct = correctness_reward(resp, gt)
        r_format = format_reward(resp)
        rewards.append(r_correct + r_format)

        if insight_scorer and alpha > 0:
            insight = extract_phase(resp, "Phase 4", "Phase 5")
            if insight and len(insight) > 20:
                insight_items.append((questions[i], options_list[i], insight))
                insight_indices.append(i)

    # Slow: insight scores (batch API)
    if insight_items:
        insight_scores = insight_scorer.score_batch(insight_items)
        for idx, score in zip(insight_indices, insight_scores):
            rewards[idx] += alpha * score

    return rewards
