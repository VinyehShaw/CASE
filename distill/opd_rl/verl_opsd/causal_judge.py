"""
causal_judge.py — LLM semantic causal evaluation for insight→answer.

Evaluates whether the student's insight provides evidence that supports the answer.
Two modes:
1. Local mode: uses local Qwen3-VL-8B model on a single GPU for inference
2. API mode: uses qwen3.7-plus via DashScope API with concurrent calls

Judge prompt evaluates:
- Does the insight identify key findings relevant to the question?
- Does the insight logically lead to the chosen answer?
- Is there a causal chain from evidence → insight → answer?

Returns: causal_score (0.0-1.0) where 1.0 = strong evidence, 0.0 = no evidence
"""

import os
import json
import time
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

# ─── Judge Log Storage ───

_JUDGE_LOG_PATH = os.environ.get("JUDGE_LOG_PATH", "judge_log.jsonl")
# Retry ceiling for judge API calls. 429s are frequent under high reward
# concurrency but almost always clear within a retry or two; backoff is
# 2**(attempt+1) seconds for 429, flat 2s otherwise.
_MAX_JUDGE_ATTEMPTS = int(os.environ.get("JUDGE_MAX_ATTEMPTS", "4"))

# Tunable judge generation (module-level; override before scoring).
# JUDGE_EXTRA_BODY: optional extra_body for the API call, e.g.
#   {"enable_thinking": False} to turn off reasoning on reasoning models
#   (faster + no reasoning tokens eating the budget).
# JUDGE_MAX_TOKENS_PER_SAMPLE: output token budget per sample in a batch call.
JUDGE_EXTRA_BODY = (
    {"enable_thinking": False}
    if os.environ.get("JUDGE_DISABLE_THINKING", "0") == "1" else None
)
JUDGE_MAX_TOKENS_PER_SAMPLE = int(os.environ.get("JUDGE_MAX_TOKENS_PER_SAMPLE", "60"))

def _log_judge_entry(question, options, ground_truth, student_answer,
                     student_insight, raw_response, parsed_score, judge_type="api"):
    """Append a judge evaluation entry to JSONL log file."""
    try:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_type": judge_type,
            "question": str(question)[:300],
            "ground_truth": str(ground_truth),
            "student_answer": str(student_answer)[:200],
            "student_insight": str(student_insight)[:500],
            "raw_response": str(raw_response)[:2000],
            "score": parsed_score,
        }
        log_dir = os.path.dirname(_JUDGE_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_JUDGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ─── Judge Prompt ───

CAUSAL_JUDGE_PROMPT = """You are a STRICT expert medical evaluator. Your task is to evaluate whether the student's insight (Phase 4: Moderator Synthesis) provides evidence that logically supports their answer to a clinical question.

## Question
{question}

## Options
{options}

## Correct Answer
{ground_truth}

## Student's Answer
{student_answer}

## Student's Insight (Phase 4: Moderator Synthesis)
{student_insight}

## Evaluation Criteria
Rate the causal evidence quality from 0.0 to 0.9:

1. **Evidence Identification (0-0.2)**: Does the insight identify SPECIFIC imaging findings (not vague descriptions)?
2. **Logical Reasoning (0-0.2)**: Does the insight build a COMPLETE logical chain from findings → interpretation → answer?
3. **Answer Support (0-0.2)**: Does the insight SPECIFICALLY support the chosen answer with evidence?
4. **Causal Completeness (0-0.3)**: Is the causal chain complete with NO gaps between evidence and conclusion?

## CRITICAL CALIBRATION RULES
- Most insights should score between 0.2 and 0.5. This is the NORMAL range.
- Scores above 0.6 are RARE and require exceptional, specific, complete reasoning.
- Do NOT give high scores just because the insight mentions the correct answer or relevant keywords.
- Vague or generic descriptions like "abnormal finding" or "significant change" should score below 0.3.
- A score of 0.7+ requires: specific imaging findings + complete logical chain + no gaps + direct answer support.
- A score of 0.9 is reserved for near-perfect insight that a senior radiologist would endorse without reservation.

## Score interpretation (STRICT):
- 0.0-0.1: No evidence, random answer
- 0.2-0.3: Vague mention, no specific findings or incomplete reasoning (MOST COMMON)
- 0.3-0.4: Some findings but reasoning is incomplete or has gaps
- 0.4-0.5: Moderate evidence, partial causal chain (GOOD)
- 0.5-0.6: Above average, most elements present but not fully connected
- 0.6-0.7: Strong evidence, minor gaps (RARE)
- 0.7-0.8: Very strong, nearly complete causal chain (VERY RARE)
- 0.8-0.9: Exceptional, complete and specific (EXTREMELY RARE)

Output ONLY a single JSON object:
{{"score": <float>, "evidence": "<brief>", "reasoning": "<brief>", "support": "<brief>", "completeness": "<brief>"}}"""


# ─── Local Model Judge (single GPU) ───

class LocalCausalJudge:
    """Use local Qwen3-VL-8B model to evaluate causal evidence quality.
    
    Runs on a single GPU, processes one sample at a time.
    No API calls needed.
    """

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = device
        self.model_path = model_path or os.environ.get(
            "JUDGE_MODEL_PATH",
            "/mnt/faster_nas/benyou.mys/medixrl/checkpoints/sft_mm_v2/checkpoint-500"
        )

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print(f"[causal_judge] Loading local model from {self.model_path} on {self.device}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        print("[causal_judge] Model loaded")

    @torch.no_grad if False else lambda f: f
    def score_single(self, question: str, options: dict, ground_truth: str,
                      student_answer: str, student_insight: str) -> dict:
        """Score a single sample using local model.

        Returns: {"score": float, "reasoning": str, ...}
        """
        import torch

        self._ensure_loaded()

        # Build options text
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)

        prompt = CAUSAL_JUDGE_PROMPT.format(
            question=question[:500],
            options=options_text[:500],
            ground_truth=ground_truth,
            student_answer=student_answer or "N/A",
            student_insight=student_insight[:1000] if student_insight else "No insight provided",
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        response = self.processor.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        result = self._parse_response(response)
        _log_judge_entry(question, options, ground_truth, student_answer,
                         student_insight, response, result.get("score", 0.5), "local")
        return result

    def score_batch(self, items: List[dict]) -> List[dict]:
        """Score multiple samples sequentially.

        items: list of {question, options, ground_truth, student_answer, student_insight}
        Returns: list of score dicts
        """
        results = []
        for i, item in enumerate(items):
            result = self.score_single(**item)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"[causal_judge] Processed {i+1}/{len(items)}")
        return results

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response to extract score."""
        import re

        # Try to parse JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if "score" in result:
                    result["score"] = max(0.0, min(1.0, float(result["score"])))
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract number from response
        numbers = re.findall(r'(\d+\.?\d*)', response)
        if numbers:
            score = max(0.0, min(1.0, float(numbers[0])))
        else:
            score = 0.5

        return {
            "score": score,
            "raw_response": response[:200],
        }


# ─── API Judge (concurrent) ───

class APICausalJudge:
    """Use API (qwen3.7-plus) to evaluate causal evidence quality.
    
    Supports concurrent calls for faster batch processing.
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "qwen3.7-plus", max_concurrent: int = 150):
        self.semaphore = Semaphore(max_concurrent)
        from openai import OpenAI
        import urllib.request, json as _json

        # Always check for local vLLM judge server first
        local_judge_url = None
        for port in (8001, 8000):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                local_judge_url = f"http://localhost:{port}/v1"
                break
            except Exception:
                pass

        if local_judge_url:
            self.base_url = local_judge_url
            self.api_key = "dummy"
            # Query the server for the correct model name
            try:
                resp = urllib.request.urlopen(f"{local_judge_url}/models", timeout=5)
                models = _json.loads(resp.read()).get("data", [])
                self.model = models[0]["id"] if models else model
            except Exception:
                self.model = model
            print(f"[causal_judge] Using local vLLM judge server at {self.base_url} (model={self.model})")
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
            self.model = model
            if not self.api_key or self.api_key == "dummy":
                raise ValueError("No API key and no local judge server available")
            print(f"[causal_judge] API judge initialized (model={self.model}, base_url={self.base_url})")

        import httpx
        self.client = OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url if self.base_url else None,
            timeout=300.0,
            max_retries=3,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=500, max_keepalive_connections=500)
            ),
        )

    def score_batch_api(self, items: List[dict], batch_size: int = 4, max_workers: int = 500) -> List[dict]:
        """Score multiple samples in batches via API with full reasoning.

        Each batch sends N samples in one API call, with full evaluation criteria
        and per-sample reasoning output (evidence/reasoning/support/completeness).
        """
        results = [None] * len(items)

        batches = []
        for i in range(0, len(items), batch_size):
            batches.append((i, items[i:i+batch_size]))

        def _score_one_batch(batch_start, batch_items):
            prompt = self._build_batch_prompt(batch_items)

            extra = {"extra_body": JUDGE_EXTRA_BODY} if JUDGE_EXTRA_BODY else {}
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=len(batch_items) * JUDGE_MAX_TOKENS_PER_SAMPLE,
                        **extra,
                    )
                    text = response.choices[0].message.content.strip()
                    scores = self._parse_batch_response(text, len(batch_items))
                    for idx, item in enumerate(batch_items):
                        _log_judge_entry(
                            item.get("question", ""), item.get("options", {}),
                            item.get("ground_truth", ""), item.get("student_answer", ""),
                            item.get("student_insight", ""),
                            scores[idx].get("raw", text) if idx < len(scores) else text,
                            scores[idx].get("score", 0.5) if idx < len(scores) else 0.5,
                            "api_batch"
                        )
                    return batch_start, scores
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        return batch_start, [{"score": 0.5}] * len(batch_items)
            return batch_start, [{"score": 0.5}] * len(batch_items)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        actual_workers = min(max_workers, len(batches)) if len(batches) > 0 else 1
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_score_one_batch, bi, bt): bi for bi, bt in batches}
            for future in as_completed(futures):
                start_idx, scores = future.result()
                for j, score in enumerate(scores):
                    if start_idx + j < len(results):
                        results[start_idx + j] = score

        return results

    def _build_batch_prompt(self, batch_items):
        """Build batch prompt with full evaluation criteria and per-sample reasoning."""
        parts = [
            "You are an expert medical evaluator. Evaluate each sample's insight (Phase 4: Moderator Synthesis) for causal evidence quality.\n",
            "IMPORTANT: Evaluate each sample INDEPENDENTLY. Different samples will likely deserve different scores. Do NOT give the same score to all samples.\n\n",
            "## Evaluation Criteria\n",
            "Rate each from 0.0 to 1.0:\n",
            "1. **Evidence Identification (0-0.3)**: Does the insight identify specific imaging findings relevant to the question?\n",
            "2. **Logical Reasoning (0-0.3)**: Does the insight build a logical chain from findings to interpretation to answer?\n",
            "3. **Answer Support (0-0.2)**: Does the insight specifically support the chosen answer (or correct answer if wrong)?\n",
            "4. **Causal Completeness (0-0.2)**: Is the causal chain complete? Are there gaps between evidence and conclusion?\n\n",
            "Score interpretation:\n",
            "- 0.0-0.2: No evidence, random answer\n",
            "- 0.2-0.4: Weak evidence, incomplete reasoning\n",
            "- 0.4-0.6: Moderate evidence, some gaps\n",
            "- 0.6-0.8: Strong evidence, minor gaps\n",
            "- 0.8-1.0: Excellent evidence, complete causal chain\n\n",
            "For EACH sample, output a JSON object on its own line. Keep the reason to ONE short sentence:\n",
            '--- Sample N ---\n{"score": <float>, "reason": "<one short sentence>"}\n\n',
        ]

        for idx, item in enumerate(batch_items):
            options_text = "\n".join(f"{k}. {v}" for k, v in item.get("options", {}).items()) if isinstance(item.get("options"), dict) else str(item.get("options", ""))
            parts.append(f"### Sample {idx+1}\n")
            parts.append(f"Question: {str(item.get('question',''))[:300]}\n")
            parts.append(f"Options: {options_text[:300]}\n")
            parts.append(f"Correct Answer: {item.get('ground_truth','')}\n")
            parts.append(f"Student Answer: {item.get('student_answer','N/A')}\n")
            parts.append(f"Student Insight: {str(item.get('student_insight',''))[:800]}\n\n")

        return "".join(parts)

    def _parse_batch_response(self, text: str, expected_count: int) -> List[dict]:
        """Parse batch response: extract per-sample JSON with reasoning."""
        import re
        scores = []
        for i in range(expected_count):
            pattern = rf'---\s*Sample\s*{i+1}\s*---\s*(\{{[^}}]+\}})'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    score = max(0.0, min(1.0, float(result.get("score", 0.5))))
                    result["score"] = score
                    result["raw"] = match.group(1)
                    scores.append(result)
                except (json.JSONDecodeError, ValueError):
                    scores.append({"score": 0.5, "raw": match.group(1)[:200]})
            else:
                pattern2 = rf'Sample\s*{i+1}.*?"score"\s*:\s*([\d.]+)'
                match2 = re.search(pattern2, text, re.IGNORECASE)
                if match2:
                    score = max(0.0, min(1.0, float(match2.group(1))))
                    scores.append({"score": score, "raw": ""})
                else:
                    scores.append({"score": 0.5, "raw": ""})
        return scores

    def score_single(self, question: str, options: dict, ground_truth: str,
                     student_answer: str, student_insight: str) -> dict:
        """Score a single sample using API."""
        import time as _time
        import os as _os
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)

        _thinking = _os.environ.get("JUDGE_THINKING", "1") == "1"

        if _thinking:
            prompt = CAUSAL_JUDGE_PROMPT.format(
                question=question[:500],
                options=options_text[:500],
                ground_truth=ground_truth,
                student_answer=student_answer or "N/A",
                student_insight=student_insight[:1000] if student_insight else "No insight provided",
            )
            _max_tokens = 400
        else:
            prompt = f"""Evaluate the causal evidence quality of this medical insight.

Question: {question[:300]}
Options: {options_text[:300]}
Correct Answer: {ground_truth}
Student Answer: {student_answer or "N/A"}
Student Insight: {student_insight[:500] if student_insight else "No insight provided"}

Score with ONE decimal place based on these levels:
- 0.0-0.1: No relevant evidence, random answer
- 0.2-0.3: Vague mention, no specific findings
- 0.4-0.5: Some relevant findings but incomplete reasoning
- 0.6-0.7: Good evidence with minor gaps in causal chain
- 0.8-0.9: Strong evidence, clear and specific causal chain
- 1.0: Perfect evidence, complete causal chain from findings to answer

Output ONLY: <score> - <one sentence reason>
Example: 0.3 - The insight mentions liver abnormalities but doesn't connect findings to the answer.
Example: 0.7 - Good identification of fat fraction but missing temporal comparison."""
            _max_tokens = 100

        _dbg_t0 = _time.time()
        for attempt in range(_MAX_JUDGE_ATTEMPTS):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=_max_tokens,
                    )
                    _dbg_elapsed = _time.time() - _dbg_t0
                    if _dbg_elapsed > 10:
                        print(f"[JUDGE_DEBUG] call took {_dbg_elapsed:.1f}s (attempt {attempt})", flush=True)
                    text = response.choices[0].message.content.strip()
                    result = self._parse_response(text)
                    _log_judge_entry(question, options, ground_truth, student_answer,
                                     student_insight, text, result.get("score", 0.5), "api")
                    return result
                except Exception as e:
                    _err = str(e)[:100]
                    print(f"[JUDGE_DEBUG] error attempt {attempt}: {_err}", flush=True)
                    if "429" in str(e) or "rate" in str(e).lower():
                        time.sleep(2 ** (attempt + 1))
                    elif attempt < _MAX_JUDGE_ATTEMPTS - 1:
                        time.sleep(2)
                    else:
                        return {"score": 0.5, "error": str(e)[:100]}
        return {"score": 0.5}

    def score_batch(self, items: List[dict]) -> List[dict]:
        """Score multiple samples concurrently."""
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.semaphore._value) as executor:
            futures = {
                executor.submit(self.score_single, **item): idx
                for idx, item in enumerate(items)
            }
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {"score": 0.5}

        return results

    def _parse_response(self, response: str) -> dict:
        """Parse API response."""
        import re
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if "score" in result:
                    result["score"] = max(0.0, min(1.0, float(result["score"])))
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

        numbers = re.findall(r'(\d+\.?\d*)', response)
        score = max(0.0, min(1.0, float(numbers[0]))) if numbers else 0.5
        return {"score": score, "raw_response": response[:200]}


# ─── Factory: auto-select local or API mode ───

import threading
_judge = None
_judge_lock = threading.Lock()

def get_causal_judge(mode: str = "auto", **kwargs):
    """Get or create a causal judge (thread-safe singleton).
    
    mode: "local" = use local 8B model
          "api" = use API
          "auto" = use API if valid key, otherwise local
    """
    global _judge
    if _judge is not None:
        return _judge

    with _judge_lock:
        if _judge is not None:  # Double-check after acquiring lock
            return _judge

        if mode == "api" or (mode == "auto" and _has_valid_api_key()):
            try:
                _judge = APICausalJudge(
                    api_key=kwargs.get("api_key", ""),
                    base_url=kwargs.get("base_url", ""),
                    model=kwargs.get("judge_model", "qwen3.7-plus"),
                    max_concurrent=kwargs.get("judge_max_concurrent", 150),
                )
                return _judge
            except Exception as e:
                print(f"[causal_judge] API mode failed: {e}, falling back to local")

        _judge = LocalCausalJudge(
            model_path=kwargs.get("judge_model_path"),
            device=kwargs.get("judge_device", "cuda:0"),
        )
        return _judge


def _has_valid_api_key():
    """Check if a valid API key OR local judge server is available."""
    # Check for local vLLM judge server
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8001/health", timeout=2)
        return True  # Local judge server is running
    except Exception:
        pass
    # Check for API key
    key = os.environ.get("OPENAI_API_KEY", "")
    return key and not key.startswith("dummy") and len(key) > 20


# ─── Convenience function for reward integration ───

def evaluate_causal_evidence(question: str, options: dict, ground_truth: str,
                              student_answer: str, student_insight: str,
                              mode: str = "auto", **kwargs) -> float:
    """One-shot evaluation: returns causal score 0.0-1.0.
    
    This is the main entry point for reward functions.
    """
    judge = get_causal_judge(mode=mode, **kwargs)
    result = judge.score_single(
        question=question,
        options=options,
        ground_truth=ground_truth,
        student_answer=student_answer,
        student_insight=student_insight,
    )
    return result.get("score", 0.5)


def evaluate_causal_evidence_batch(items: List[dict], mode: str = "auto", **kwargs) -> List[float]:
    """Batch evaluation: returns list of causal scores 0.0-1.0.

    Uses score_batch_api to reduce API calls (8 samples per call).
    Falls back to score_single for LocalCausalJudge (no batch method).

    Kwargs:
        judge_batch_size: samples per API call (default 4)
        judge_max_workers: concurrent API workers (default 48)
    """
    batch_size = kwargs.pop("judge_batch_size", 4)
    max_workers = kwargs.pop("judge_max_workers", 48)

    judge = get_causal_judge(mode=mode, **kwargs)

    if hasattr(judge, 'score_batch_api'):
        results = judge.score_batch_api(items, batch_size=batch_size, max_workers=max_workers)
        return [r.get("score", 0.5) if r else 0.5 for r in results]
    else:
        # Fallback for LocalCausalJudge
        return [
            judge.score_single(
                question=item.get("question", ""),
                options=item.get("options", {}),
                ground_truth=item.get("ground_truth", ""),
                student_answer=item.get("student_answer", ""),
                student_insight=item.get("student_insight", ""),
            ).get("score", 0.5)
            for item in items
        ]
