#!/usr/bin/env python3
"""Skill-memory mock tool for verl ToolAgentLoop RL (medboard module).

Two named tools from one class:
- recall_skills: deterministic tag/keyword match against a static skill store
  (JSONL built by build_skill_store.py). Lookup is pure Python over a frozen
  store, so replays are deterministic without a per-patient cache.
- write_skill: accepts the skill text and returns a fixed acknowledgement
  (no persistence during RL rollout).
"""

import json
import os
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

_STORES: dict = {}
_STORE_LOCK = threading.Lock()

TOP_K = 3


def _load_store(store_path: str) -> list:
    with _STORE_LOCK:
        if store_path not in _STORES:
            skills = []
            with open(store_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        skills.append(json.loads(line))
            _STORES[store_path] = skills
        return _STORES[store_path]


def match_skills(store: list, organ: str, keywords: list, top_k: int = TOP_K) -> list:
    """Score skills by organ match + keyword overlap. Deterministic ordering."""
    organ = (organ or "").lower()
    kw = set(k.lower() for k in keywords if k)
    scored = []
    for i, sk in enumerate(store):
        score = 0.0
        if organ and sk.get("organ", "").lower() == organ:
            score += 2.0
        elif organ and sk.get("organ", "").lower() == "general":
            score += 0.5
        sk_kw = set(k.lower() for k in sk.get("keywords", []))
        score += 0.5 * len(kw & sk_kw)
        if sk.get("source") == "failure_reflection":
            score += 0.3  # hard-won lessons rank above generic guidance
        if score > 0:
            scored.append((-score, i, sk))
    scored.sort()
    return [sk for _, _, sk in scored[:top_k]]


def format_skills(skills: list) -> str:
    if not skills:
        return "No matching skills found in the skill library."
    parts = [f"[Recalled {len(skills)} skill(s) from the skill library]"]
    for sk in skills:
        tag = " [LESSON FROM FAILURE]" if sk.get("source") == "failure_reflection" else ""
        parts.append(f"### Skill: {sk.get('patient_type', 'unknown type')}{tag}")
        parts.append(f"Organ: {sk.get('organ', '?')} | Keywords: {', '.join(sk.get('keywords', []))}")
        parts.append(f"Guidance: {sk.get('guidance', '')}")
        if sk.get("pitfalls"):
            parts.append(f"Pitfalls: {sk['pitfalls']}")
        parts.append("")
    return "\n".join(parts)


class SkillMemoryMockTool(BaseTool):
    """Replay recall_skills from a static store; write_skill is a no-op ack."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.store = _load_store(config["store_path"])
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instances[instance_id] = str(create_kwargs.get("patient_id", ""))
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        params = parameters or {}

        if self.name == "recall_skills":
            organ = str(params.get("organ", ""))
            keywords = params.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            skills = match_skills(self.store, organ, keywords)
            return ToolResponse(text=format_skills(skills)), 0.0, {}

        if self.name == "write_skill":
            patient_type = str(params.get("patient_type", "")).strip()
            guidance = str(params.get("guidance", "")).strip()
            if not patient_type or not guidance:
                return ToolResponse(
                    text="Error: write_skill requires non-empty 'patient_type' and 'guidance'."), 0.0, {}
            if os.environ.get("MEDBOARD_LIVE_SKILL", "0") == "1":
                kws = params.get("keywords") or []
                if isinstance(kws, str):
                    kws = [k.strip() for k in kws.split(",") if k.strip()]
                skill = {
                    "skill_id": f"mb_rt_{len(self.store) + 1:04d}",
                    "organ": (str(params.get("organ", "general")).strip().lower() or "general"),
                    "keywords": kws,
                    "patient_type": patient_type,
                    "guidance": guidance[:600],
                    "pitfalls": str(params.get("pitfalls", ""))[:400],
                    "source": str(params.get("source", "failure_reflection")),
                }
                with _STORE_LOCK:
                    self.store.append(skill)
            return ToolResponse(text=(
                f"Skill recorded for patient type '{patient_type[:80]}'. "
                "It will be available via recall_skills for similar future patients.")), 0.0, {}

        return ToolResponse(text=f"Unknown tool: {self.name}"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
