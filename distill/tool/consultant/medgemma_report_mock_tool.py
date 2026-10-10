#!/usr/bin/env python3
"""MedGemma report mock tool for verl ToolAgentLoop RL (medboard module).

Reads the unified NAS medgemma_0820 source (manifest + raw report JSONs) with
online compact formatting (three fixed fields) and ICD-10 masking — identical
logic to the recording-side medboard_tool.get_report, so RL rollouts see the
same report text/images the SFT trajectories were recorded with. Attached
images are resized (longest edge 512) to keep the RL token budget safe.
"""

import base64
import io
import json
import os
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

MEDGEMMA_IMAGE_BASE = os.environ.get(
    "MEDGEMMA_IMAGE_BASE", "/mnt/faster_nas/benyou.mys/dataset/medgemma_0820")
MEDGEMMA_MANIFEST = os.path.join(MEDGEMMA_IMAGE_BASE, "manifest.json")

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

_NOT_FOUND = {
    "list_available_reports": "No MedGemma reports available for this patient.",
    "get_report": "Error: no {report_type} report available for organ '{organ}'.",
}


def _load_manifest() -> dict:
    with _CACHE_LOCK:
        if "manifest" not in _CACHE:
            try:
                with open(MEDGEMMA_MANIFEST) as f:
                    _CACHE["manifest"] = json.load(f)
            except Exception:
                _CACHE["manifest"] = {}
        return _CACHE["manifest"]


def _format_report(doc: dict, rt: str) -> str:
    """Compact three-field formatting, same as build_report_cache FORMATTERS."""
    try:
        from distill.tool.consultant.build_report_cache import (
            FORMATTERS, mask_diagnosis_codes, _clip)
        return FORMATTERS[rt](doc)
    except Exception:
        # minimal inline fallback if import fails
        header = f"[MedGemma {rt} | {doc.get('anatomy', '?')} | {doc.get('modality', '?')}]"
        body = doc.get(rt) or (doc.get("report") or {}).get("comparison_summary") or doc.get("interval_change") or ""
        return header + "\n" + str(body)[:400]


def _report_images(patient_id: str, organ: str, max_imgs: int = 2):
    """(b64_list, label_list): montage/overview preferred, longest-edge-512."""
    man = _load_manifest()
    paths = []
    for visit in ("visit_2", "visit_3"):
        paths += man.get(f"{patient_id}::{organ}::{visit}", [])
    if not paths:
        return [], []
    paths.sort(key=lambda p: 0 if ("montage" in p or "overview" in p) else 1)
    out, labels = [], []
    from PIL import Image
    for p in paths[:max_imgs]:
        fp = os.path.join(MEDGEMMA_IMAGE_BASE, p)
        if not os.path.exists(fp):
            continue
        try:
            img = Image.open(fp).convert("RGB")
            w, h = img.size
            if max(w, h) > 512:
                s = 512 / max(w, h)
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
            labels.append("medgemma " + organ)
        except Exception:
            continue
    return out, labels


class MedGemmaReportMockTool(BaseTool):
    """NAS-manifest-backed replay of MedGemma reports (compact + images)."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self._manifest = _load_manifest()
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instances[instance_id] = str(create_kwargs.get("patient_id", ""))
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        patient_id = self._instances.get(instance_id, "")
        params = parameters or {}

        if not patient_id:
            return ToolResponse(text="Error: No patient ID set"), 0.0, {}

        if self.name == "list_available_reports":
            organs = ("brain", "heart", "liver", "pancreas", "wholebody")
            lines = []
            for organ in organs:
                rp = self._manifest.get(f"{patient_id}::{organ}::reports", [])
                if rp:
                    rts = sorted(p.split("__")[-1][:-len("_report.json")] for p in rp)
                    lines.append(f"- {organ} ({', '.join(rts)})")
            if not lines:
                return ToolResponse(text=_NOT_FOUND["list_available_reports"]), 0.0, {}
            return ToolResponse(text="[Available MedGemma reports]\n" + "\n".join(lines)), 0.0, {}

        if self.name == "get_report":
            organ = str(params.get("organ", "")).strip().lower()
            rt = str(params.get("report_type", "")).strip().lower()
            rp = self._manifest.get(f"{patient_id}::{organ}::reports", [])
            path = next((p for p in rp if p.endswith(f"__{rt}_report.json")), None)
            if path is None:
                return ToolResponse(
                    text=_NOT_FOUND["get_report"].format(report_type=rt, organ=organ)), 0.0, {}
            try:
                with open(os.path.join(MEDGEMMA_IMAGE_BASE, path)) as f:
                    doc = json.load(f)
            except Exception as e:
                return ToolResponse(text=f"Error reading report: {e}"), 0.0, {}
            text = _format_report(doc, rt)
            b64s, labels = _report_images(patient_id, organ)
            return ToolResponse(text=text, image=b64s, image_labels=labels), 0.0, {}

        return ToolResponse(text=f"Unknown tool: {self.name}"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
