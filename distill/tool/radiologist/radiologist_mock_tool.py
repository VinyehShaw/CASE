#!/usr/bin/env python3
"""MedPanel v2 mock tool for verl ToolAgentLoop RL.

Replays precomputed tool results (text + images) built offline by
build_medpanel_v2_cache.py, keyed by (patient_id, tool_name, modality, visit).
The patient_id arrives per-trajectory via tools_kwargs create_kwargs in the RL parquet.

verl's ToolAgentLoop inserts a single {"type": "image"} placeholder per tool
message, so images are returned as a single PIL image per execute call.

Requires PYTHONPATH to include the ezdistill repo root.
"""

import json
import os
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

MAX_IMAGES = 6
IMG_WIDTH = 512
IMG_HEIGHT = 512

NOT_FOUND_TEXT = {
    "list_available_images": "No imaging data available for this patient.",
    "view_image": "Error: modality '{modality}' not available for {visit}",
    "compare_visits": "Error: {modality} not available for both visits",
}


def _load_cache(cache_path: str) -> dict:
    with _CACHE_LOCK:
        if cache_path not in _CACHE:
            with open(cache_path) as f:
                _CACHE[cache_path] = json.load(f)
        return _CACHE[cache_path]


def _stack_images(image_entries: list, cache_dir: str):
    """Load PNGs and stack vertically into one PIL image."""
    from PIL import Image

    imgs = []
    labels = []
    for e in image_entries[:MAX_IMAGES]:
        p = os.path.join(cache_dir, e["path"])
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        if img.width > IMG_WIDTH:
            h = int(img.height * IMG_WIDTH / img.width)
            img = img.resize((IMG_WIDTH, h), Image.LANCZOS)
        imgs.append(img)
        labels.append(e.get("label", ""))
    if not imgs:
        return None, []

    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + 4 * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height + 4
    canvas = canvas.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return canvas, labels


class MedPanelV2MockTool(BaseTool):
    """Dict-lookup replay of the real MedPanel v2 LangGraph tools."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.cache_path = config["cache_path"]
        self.cache_dir = os.path.dirname(self.cache_path)
        self.include_images = config.get("include_images", True)
        self.cache = _load_cache(self.cache_path)
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

        if self.name == "list_available_images":
            key = f"{patient_id}::list_available_images"
        elif self.name == "view_image":
            modality = params.get("modality", "")
            visit = params.get("visit", "")
            key = f"{patient_id}::view_image::{modality}::{visit}"
        elif self.name == "compare_visits":
            modality = params.get("modality", "")
            key = f"{patient_id}::compare_visits::{modality}"
        else:
            return ToolResponse(text=f"Unknown tool: {self.name}"), 0.0, {}

        entry = self.cache.get(key)
        if entry is None:
            modality = params.get("modality", "")
            visit = params.get("visit", "")
            text = NOT_FOUND_TEXT.get(
                self.name, "Error: entry not found"
            ).format(modality=modality, visit=visit)
            return ToolResponse(text=text), 0.0, {}

        text = entry["text"]
        image = None
        if self.include_images and entry.get("images"):
            stacked, labels = _stack_images(entry["images"], self.cache_dir)
            if stacked is not None:
                image = [stacked]
                text += f"\n  [Images: {', '.join(labels)}]"

        return ToolResponse(text=text, image=image), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
