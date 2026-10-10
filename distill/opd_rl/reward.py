#!/usr/bin/env python3
"""
reward.py — Reward functions for OPD-RL GRPO training.

Two reward components:
  1. correctness_reward (sparse): +1 correct, -1 wrong, -2 invalid
  2. quality_reward (dense): LLM judge score (0-1), via API

Total: R_total = R_sparse + alpha * R_dense
"""

import hashlib
import json
import os
import re
import time
from threading import Semaphore

from openai import OpenAI


def extract_answer(text):
    """Extract the answer from a model response.

    Supports:
      - Single answer: "Answer: B" → "B"
      - Multi answer: "Answer: A,B" → "A,B" (sorted)
      - Answers A-I (not just A-E)

    When multiple Answer lines exist (e.g. "Answer: A,B,C" followed by "Answer: A"),
    prefer the one with more letters (the complete multi-answer).

    Returns the answer string (e.g. "B", "A,B", "F") or None.
    """
    patterns = [
        # "**Answer:** A,B,C" — requires colon to avoid matching "answer is"
        r"(?:\*\*)?[Aa]nswer[:：]?(?:\*\*)?:?\s*[:：]\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)",
        # "Answer: A,B" — requires colon
        r"(?:\*\*)?[Aa]nswer\s*[:：]\s*\*?\*?\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)",
        # "answer is A,B" — explicit "answer is" pattern
        r"(?:\*\*)?[Aa]nswer\s+is\s*\*?\*?\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)",
        # Chinese answer pattern "答案: A,B"
        r"(?:最终)?答案\s*[:：]\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)",
        # Chinese choice pattern "选择 A,B"
        r"选择\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)",
        # Letter at end of line "B" (fallback)
        r"\b([A-I](?:\s*,\s*[A-I])*)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            parsed = []
            for raw in matches:
                letters = sorted(set(l.strip().upper() for l in raw.split(",") if l.strip()))
                if letters:
                    parsed.append(letters)
            if parsed:
                parsed.sort(key=len, reverse=True)
                return ",".join(parsed[0])
    return None


class RewardCache:
    """File-based cache for quality reward scores to avoid re-judging."""

    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["score"]

    def get(self, prompt, completion):
        key = self._make_key(prompt, completion)
        return self.cache.get(key)

    def put(self, prompt, completion, score):
        key = self._make_key(prompt, completion)
        self.cache[key] = score
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "score": score}) + "\n")

    def _make_key(self, prompt, completion):
        prompt_str = json.dumps(prompt, sort_keys=True) if isinstance(prompt, (list, dict)) else str(prompt)
        content = prompt_str + str(completion)
        return hashlib.md5(content.encode()).hexdigest()


def correctness_reward(prompts, completions, **kwargs):
    """Sparse reward: +1 correct, -1 wrong, -2 invalid answer.

    kwargs must contain 'ground_truth' as a list of answer letters.
    """
    ground_truths = kwargs.get("ground_truth", [])
    rewards = []
    for i, completion in enumerate(completions):
        gt = ground_truths[i] if i < len(ground_truths) else None
        answer = extract_answer(completion)
        if answer is None:
            rewards.append(-2.0)
        elif gt and answer == gt:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def make_quality_reward(api_key, base_url, model, max_concurrent=20, cache_path=None, alpha=0.5):
    """Create a quality reward function that calls LLM judge.

    Returns a function suitable for TRL GRPOTrainer's reward_funcs.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        timeout=300,
    )
    semaphore = Semaphore(max_concurrent)
    cache = RewardCache(cache_path)

    JUDGE_PROMPT = """Score this medical discussion response on a scale of 0.0 to 1.0.

Question: {question}
Ground Truth: {gt}

Student Response:
{response}

Output ONLY a number between 0.0 and 1.0."""

    def quality_reward(prompts, completions, **kwargs):
        ground_truths = kwargs.get("ground_truth", [])
        questions = kwargs.get("question", [])
        rewards = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def score_one(idx, completion):
            prompt = prompts[idx]
            gt = ground_truths[idx] if idx < len(ground_truths) else ""
            question = questions[idx] if idx < len(questions) else ""

            cached = cache.get(prompt, completion)
            if cached is not None:
                return cached

            prompt_text = JUDGE_PROMPT.format(
                question=question[:500],
                gt=gt,
                response=completion[:2000],
            )

            with semaphore:
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt_text}],
                            temperature=0.0,
                            max_tokens=10,
                        )
                        score_text = resp.choices[0].message.content.strip()
                        score = float(score_text)
                        score = max(0.0, min(1.0, score))
                        cache.put(prompt, completion, score)
                        return score
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            time.sleep(2 ** (attempt + 1))
                        else:
                            return 0.5
            return 0.5

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(score_one, i, comp): i
                for i, comp in enumerate(completions)
            }
            results = [0.5] * len(completions)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [r * alpha for r in results]

    return quality_reward


def make_combined_reward(api_key, base_url, model, alpha=0.5, max_concurrent=20,
                         cache_path=None, correct_reward=1.0, wrong_reward=-1.0,
                         invalid_reward=-2.0, use_dense=True):
    """Create a combined reward function: R_total = R_sparse + alpha * R_dense."""

    quality_fn = make_quality_reward(
        api_key, base_url, model, max_concurrent, cache_path, 1.0
    ) if use_dense else None

    def combined_reward(prompts, completions, **kwargs):
        ground_truths = kwargs.get("ground_truth", [])

        sparse_rewards = []
        for i, completion in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else None
            answer = extract_answer(completion)
            if answer is None:
                sparse_rewards.append(invalid_reward)
            elif gt and answer == gt:
                sparse_rewards.append(correct_reward)
            else:
                sparse_rewards.append(wrong_reward)

        if quality_fn and use_dense:
            dense_rewards = quality_fn(prompts, completions, **kwargs)
        else:
            dense_rewards = [0.0] * len(completions)

        return [s + alpha * d for s, d in zip(sparse_rewards, dense_rewards)]

    return combined_reward
