"""
Vista3D LangGraph Agent — Pre-computed segmentation tool calling.

Architecture:
    START → llm (decides which organ to query via tool_calls)
          → tools (executes vista3d_segment, returns pre-computed volume)
          → llm (analyzes results, may call more tools or answer)
          → END (when no more tool_calls)

The LLM uses OpenAI native function calling (tools parameter), NOT regex parsing.
Tool results come from pre-computed Vista3D segmentation (offline nii.gz files).

Trajectory format: standard OpenAI messages with tool_calls → SFT/RL compatible.
"""

import os
import json
import base64
import threading
from typing import TypedDict, Optional, Annotated
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from distill.tool.anatomist.anatomist_label_map import (
    resolve_organ,
    get_label_index,
    get_organ_name,
    compute_volumes_with_spacing,
    VISTA3D_LABEL_MAP,
    STATION_REGIONS,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_VISTA3D_DIR = os.environ.get("VISTA3D_DIR", "/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d")
DEFAULT_VISTA3D_DIR_V2 = os.environ.get("VISTA3D_DIR_V2", "/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2")
DEFAULT_STATIONS = ["dixon_4_W", "dixon_8_W", "dixon_12_W", "dixon_16_W", "dixon_20_W", "dixon_24_W"]

SYSTEM_PROMPT = """You are a medical imaging assistant analyzing whole-body MRI segmentation data.

The patient has TWO MRI visits: V1 (baseline) and V2 (follow-up). Each visit was scanned in multiple body stations (dixon_4_W=head/neck, dixon_8_W=thorax, dixon_12_W=abdomen, dixon_16_W=pelvis, etc.). The same organ may appear in multiple adjacent stations.

TOOLS:
1. vista3d_segment(organ) — returns per-station volume breakdown for an organ (from V1)
2. vista3d_compare_stations(organ) — compares an organ between dixon_8_W and dixon_12_W within V1
3. vista3d_compare_visits(organ) — compares an organ's volume between V1 (baseline) and V2 (follow-up), returns volume change + overlay images from both visits

RULES:
1. Call vista3d_compare_visits for AT MOST 1-3 most relevant organs. Do NOT call tools for every organ.
2. Do NOT invent or hallucinate any volume numbers.
3. After getting tool results, write your FINAL response containing ALL 5 phases below.

EFFICIENCY: Each tool call takes time. Limit yourself to 1-3 tool calls total. If the question mentions a specific organ (e.g., "liver"), call vista3d_compare_visits for ONLY that organ. If the question is general or involves multiple organs, pick the 1-3 most relevant organs based on the question and patient info. Never call the same tool twice for the same organ.

CRITICAL: Your final response (the text after tool results) MUST contain ALL 5 phases in this exact format. Do NOT skip any phase. Do NOT split phases across multiple messages. Write the complete 5-phase response in a single message. After writing the 5-phase response, STOP — do not call any more tools.

## Phase 1: Target Selection
Target: <organ name(s), comma-separated if multiple>
Reasoning: <why these organs are relevant to the question>

## Phase 2: Vista3D V1 Analysis
<restatement of V1 organ volumes from tool results, per-station breakdown>

## Phase 3: Vista3D V2 Analysis
<restatement of V2 organ volumes, volume change vs V1 from tool results>

## Phase 4: Insight Synthesis
<Synthesize ALL prior phases into ONE coherent clinical insight. Explicitly combine: Phase 1 (why these organs were selected), Phase 2 (V1 organ volumes from the tool), and Phase 3 (V2 organ volumes and volume change vs V1 from the tool). State clearly what the volume changes mean clinically. IMPORTANT: vista3d segmentation reflects anatomy and organ volumes only — it does NOT by itself establish a disease diagnosis. Base this insight ONLY on the volume data you actually received from the tools; do not invent findings or diagnoses the data does not support.>

## Phase 5: Final Diagnosis
Answer: <letter>
Reasoning: <MUST be derived directly from your Phase 4 insight. The answer must follow logically from Phase 4 and be consistent with it. Segmentation shows anatomy, not disease — if Phase 4 finds no volume evidence supporting any of the disease options, select "None of the above" if it is an option.>"""


# ============================================================
# State
# ============================================================

class Vista3DAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    patient_info: str
    question: str
    options: dict
    answer: Optional[str]
    correct: Optional[bool]


# ============================================================
# Vista3D Result Loader
# ============================================================

class Vista3DResultLoader:
    """Loads pre-computed Vista3D segmentation results from nii.gz files.

    Supports per-station volume breakdown, slice-level organ extraction,
    and V1/V2 visit comparison for tracking organ volume changes over time.
    """

    # Key abdominal stations with overlapping organs worth comparing
    COMPARISON_STATIONS = ["dixon_8_W", "dixon_12_W"]

    def __init__(self, vista3d_dir: str = DEFAULT_VISTA3D_DIR, vista3d_dir_v2: str = None):
        self.vista3d_dir = vista3d_dir
        self.vista3d_dir_v2 = vista3d_dir_v2 or DEFAULT_VISTA3D_DIR_V2
        self._volume_cache = {}      # (patient_id, organ, visit) → volume data
        self._seg_cache = {}         # (patient_id, station, visit) → (seg_array, spacing, input_array)
        self._overlay_cache = {}     # (patient_id, station, organ, slice_idx, visit) → base64 png
        self._cache_lock = threading.Lock()
        self._max_cache_size = 30   # Max entries per cache to bound memory

    def _get_patient_dir(self, visit: str, patient_id: str) -> str:
        """Return patient directory path for the specified visit.

        V1 uses plain {patient_id}/, V2 uses {patient_id}_visit3/ naming.
        """
        if visit == "v1":
            return os.path.join(self.vista3d_dir, str(patient_id))
        else:
            v2_path = os.path.join(self.vista3d_dir_v2, f"{patient_id}_visit3")
            if os.path.isdir(v2_path):
                return v2_path
            return os.path.join(self.vista3d_dir_v2, str(patient_id))

    def _discover_stations(self, patient_dir: str) -> list:
        """Auto-discover dixon_*_W station subdirs in a visit's patient folder.

        Replaces hardcoded DEFAULT_STATIONS — UKB Dixon station numbering is not
        consistent across patients/visits (some use dixon_4-24, others dixon_28-48
        or dixon_5-25). Reads whatever stations actually exist for this visit so
        patients whose two visits scanned different body regions are handled
        correctly (each visit read on its own terms).
        """
        if not os.path.isdir(patient_dir):
            return []
        stations = sorted(
            d for d in os.listdir(patient_dir)
            if d.startswith("dixon_") and d.endswith("_W")
            and os.path.isdir(os.path.join(patient_dir, d))
        )
        return stations if stations else list(DEFAULT_STATIONS)

    def _load_station_seg(self, patient_id: str, station: str, visit: str = "v1"):
        """Load segmentation + input MRI for a station, with caching."""
        cache_key = (patient_id, station, visit)
        with self._cache_lock:
            if cache_key in self._seg_cache:
                return self._seg_cache[cache_key]
            if len(self._seg_cache) > self._max_cache_size:
                self._seg_cache.clear()

        patient_dir = self._get_patient_dir(visit, patient_id)
        station_dir = os.path.join(patient_dir, station)
        seg_path = os.path.join(station_dir, "seg_trans.nii.gz")
        input_path = os.path.join(station_dir, "input.nii.gz")

        if not os.path.exists(seg_path):
            with self._cache_lock:
                self._seg_cache[cache_key] = None
            return None

        try:
            import nibabel as nib
            import numpy as np

            seg_nii = nib.load(seg_path)
            seg_data = np.asanyarray(seg_nii.dataobj)
            spacing = tuple(float(s) for s in seg_nii.header.get_zooms()[:3])

            input_data = None
            if os.path.exists(input_path):
                input_nii = nib.load(input_path)
                input_data = np.asanyarray(input_nii.dataobj)

            result = (seg_data, spacing, input_data)
            with self._cache_lock:
                self._seg_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"[WARNING] _load_station_seg failed for {patient_id}/{station}/{visit}: {e}")
            self._seg_cache[cache_key] = None
            return None

    def get_organ_volume(self, patient_id: str, organ_name: str, visit: str = "v1") -> dict:
        """Get per-station volume breakdown for an organ from a specific visit.

        Args:
            patient_id: Patient ID
            organ_name: Organ name
            visit: "v1" or "v2"

        Returns:
            {
                "organ": str,
                "found": bool,
                "volume_ml": float,            # total across all stations
                "voxel_count": int,            # total
                "stations": list[str],         # stations where organ found
                "per_station": dict,           # {station: {volume_ml, voxel_count, slice_range, max_slice}}
                "overlay_paths": list[str],    # preview PNG paths
                "visit": str,                  # which visit
            }
        """
        canonical = resolve_organ(organ_name)
        if canonical is None:
            return {
                "organ": organ_name,
                "found": False,
                "volume_ml": 0.0,
                "voxel_count": 0,
                "stations": [],
                "per_station": {},
                "overlay_paths": [],
                "visit": visit,
                "error": f"Unknown organ: {organ_name}",
            }

        label_idx = get_label_index(canonical)

        cache_key = (patient_id, canonical, visit)
        if cache_key in self._volume_cache:
            return self._volume_cache[cache_key]

        patient_dir = self._get_patient_dir(visit, patient_id)
        if not os.path.isdir(patient_dir):
            result = {
                "organ": canonical,
                "found": False,
                "volume_ml": 0.0,
                "voxel_count": 0,
                "stations": [],
                "per_station": {},
                "overlay_paths": [],
                "visit": visit,
                "error": f"Patient {patient_id} not found in {visit} data",
            }
            self._volume_cache[cache_key] = result
            return result

        total_voxel_count = 0
        total_volume_ml = 0.0
        stations_found = []
        overlay_paths = []
        per_station = {}

        for station in self._discover_stations(patient_dir):
            overlay_path = os.path.join(patient_dir, station, "preview_slices.png")
            loaded = self._load_station_seg(patient_id, station, visit)
            if loaded is None:
                continue

            try:
                import numpy as np

                seg_data, spacing, _ = loaded
                vol_data = compute_volumes_with_spacing(seg_data, label_idx, spacing)

                if vol_data["voxel_count"] > 0:
                    total_voxel_count += vol_data["voxel_count"]
                    total_volume_ml += vol_data["volume_ml"]
                    stations_found.append(station)

                    # Slice-level extraction: find which axial slices contain this organ
                    organ_mask = (seg_data == label_idx)
                    slice_counts = organ_mask.sum(axis=(0, 1))
                    active_slices = np.where(slice_counts > 0)[0]

                    slice_area_mm2 = None
                    max_slice = None
                    if len(active_slices) > 0:
                        px_area = float(spacing[0] * spacing[1])
                        slice_areas = slice_counts[active_slices] * px_area
                        max_idx = np.argmax(slice_counts[active_slices])
                        max_slice = int(active_slices[max_idx])
                        slice_area_mm2 = round(float(slice_areas[max_idx]), 1)

                    per_station[station] = {
                        "volume_ml": vol_data["volume_ml"],
                        "voxel_count": vol_data["voxel_count"],
                        "slice_range": [int(active_slices[0]), int(active_slices[-1])] if len(active_slices) > 0 else None,
                        "num_slices": int(len(active_slices)),
                        "max_slice": max_slice,
                        "max_slice_area_mm2": slice_area_mm2,
                    }

                    if os.path.exists(overlay_path):
                        overlay_paths.append(overlay_path)

            except Exception:
                continue

        result = {
            "organ": canonical,
            "found": total_voxel_count > 0,
            "volume_ml": round(total_volume_ml, 1),
            "voxel_count": total_voxel_count,
            "stations": stations_found,
            "per_station": per_station,
            "overlay_paths": overlay_paths,
            "visit": visit,
        }

        self._volume_cache[cache_key] = result
        return result

    def generate_organ_overlay(
        self,
        patient_id: str,
        organ_name: str,
        station: str,
        slice_idx: int = None,
        visit: str = "v1",
    ) -> str | None:
        """Generate a base64-encoded PNG overlay for a specific organ on a specific slice.

        The overlay shows the MRI slice with the organ segmentation highlighted in color.

        Args:
            patient_id: Patient ID
            organ_name: Canonical organ name
            station: Body station (e.g., "dixon_8_W")
            slice_idx: Axial slice index (if None, uses the slice with max organ area)
            visit: "v1" or "v2"

        Returns:
            Base64-encoded PNG string, or None if generation failed.
        """
        import numpy as np

        canonical = resolve_organ(organ_name)
        if canonical is None:
            return None

        label_idx = get_label_index(canonical)
        loaded = self._load_station_seg(patient_id, station, visit)
        if loaded is None:
            return None

        seg_data, spacing, input_data = loaded

        organ_mask = (seg_data == label_idx)
        slice_counts = organ_mask.sum(axis=(0, 1))

        if slice_idx is None:
            active = np.where(slice_counts > 0)[0]
            if len(active) == 0:
                return None
            slice_idx = int(active[np.argmax(slice_counts[active])])
        else:
            if slice_counts[slice_idx] == 0:
                return None

        cache_key = (patient_id, station, canonical, slice_idx, visit)
        if cache_key in self._overlay_cache:
            return self._overlay_cache[cache_key]

        try:
            import io
            from PIL import Image

            # Extract slice (transpose + flip for radiological orientation)
            if input_data is not None:
                slc = input_data[:, :, slice_idx].T
            else:
                slc = seg_data[:, :, slice_idx].T
            mask_slice = organ_mask[:, :, slice_idx].T
            slc = np.flipud(slc)
            mask_slice = np.flipud(mask_slice)

            # Normalize background to 0-255 grayscale RGB
            vmin, vmax = np.percentile(slc, [1, 99])
            slc = np.clip((slc - vmin) / (vmax - vmin + 1e-8), 0, 1)
            slc = (slc * 255).astype(np.float32)
            rgb = np.stack([slc, slc, slc], axis=-1)

            # Overlay organ mask in yellow with alpha=0.4
            overlay_color = np.array([255, 255, 0], dtype=np.float32)
            alpha = 0.4
            mask_idx = mask_slice
            rgb[mask_idx] = rgb[mask_idx] * (1 - alpha) + overlay_color * alpha
            rgb = rgb.astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode("utf-8")
            self._overlay_cache[cache_key] = b64
            return b64
        except Exception:
            return None

    def compare_stations(
        self,
        patient_id: str,
        organ_name: str,
        stations: list[str] = None,
        visit: str = "v1",
    ) -> dict:
        """Compare organ volumes and slices between body stations within one visit.

        Args:
            patient_id: Patient ID
            organ_name: Organ name
            stations: List of stations to compare (default: dixon_8_W vs dixon_12_W)
            visit: "v1" or "v2"

        Returns:
            {
                "organ": str,
                "comparison": {station: {volume_ml, voxel_count, slice_range, max_slice, ...}},
                "overlay_images": {station: base64_png},  # organ-highlighted overlay from max slice
                "volume_diff_ml": float,  # station[0] - station[1]
            }
        """
        if stations is None:
            stations = self.COMPARISON_STATIONS

        vol_data = self.get_organ_volume(patient_id, organ_name, visit=visit)
        canonical = vol_data["organ"]
        per_station = vol_data.get("per_station", {})

        comparison = {}
        overlay_images = {}

        for st in stations:
            if st in per_station:
                comparison[st] = per_station[st]
                overlay_b64 = self.generate_organ_overlay(patient_id, canonical, st, visit=visit)
                if overlay_b64:
                    overlay_images[st] = overlay_b64
            else:
                comparison[st] = {
                    "volume_ml": 0.0,
                    "voxel_count": 0,
                    "slice_range": None,
                    "num_slices": 0,
                    "max_slice": None,
                    "max_slice_area_mm2": None,
                }

        # Compute volume difference
        vols = [comparison.get(st, {}).get("volume_ml", 0.0) for st in stations]
        volume_diff = vols[0] - vols[1] if len(vols) >= 2 else 0.0

        return {
            "organ": canonical,
            "found": vol_data["found"],
            "total_volume_ml": vol_data["volume_ml"],
            "comparison": comparison,
            "overlay_images": overlay_images,
            "volume_diff_ml": round(volume_diff, 1),
        }

    def compare_visits(self, patient_id: str, organ_name: str) -> dict:
        """Compare an organ's volume between V1 (baseline) and V2 (follow-up).

        Generates overlay images from the max-area slice in key stations (d8, d12)
        for both visits, enabling visual comparison of organ morphology over time.

        Returns:
            {
                "organ": str,
                "v1_volume_ml": float,
                "v2_volume_ml": float,
                "volume_change_ml": float,     # v2 - v1
                "volume_change_pct": float,    # (v2-v1)/v1 * 100
                "v1_per_station": dict,
                "v2_per_station": dict,
                "overlay_images": {
                    "v1": {"dixon_8_W": base64, "dixon_12_W": base64},
                    "v2": {"dixon_8_W": base64, "dixon_12_W": base64},
                }
            }
        """
        v1_data = self.get_organ_volume(patient_id, organ_name, visit="v1")
        v2_data = self.get_organ_volume(patient_id, organ_name, visit="v2")
        canonical = v1_data["organ"]

        v1_vol = v1_data["volume_ml"]
        v2_vol = v2_data["volume_ml"]
        change_ml = round(v2_vol - v1_vol, 1)
        change_pct = round((v2_vol - v1_vol) / v1_vol * 100, 1) if v1_vol > 0 else 0.0

        # Generate overlay images: prefer d8/d12, but fall back to the organ's
        # top-2 stations by volume (prostate/bladder only exist in dixon_16_W,
        # so hardcoded stations produced zero overlays for them)
        overlay_images = {"v1": {}, "v2": {}}
        for visit, vol_data in [("v1", v1_data), ("v2", v2_data)]:
            per_st = vol_data.get("per_station", {})
            stations = [st for st in self.COMPARISON_STATIONS if st in per_st]
            if not stations and per_st:
                stations = sorted(per_st, key=lambda s: per_st[s].get("volume_ml", 0), reverse=True)[:2]
            for st in stations:
                b64 = self.generate_organ_overlay(patient_id, canonical, st, visit=visit)
                if b64:
                    overlay_images[visit][st] = b64

        return {
            "organ": canonical,
            "found": v1_data["found"] or v2_data["found"],
            "v1_volume_ml": v1_vol,
            "v2_volume_ml": v2_vol,
            "volume_change_ml": change_ml,
            "volume_change_pct": change_pct,
            "v1_per_station": v1_data.get("per_station", {}),
            "v2_per_station": v2_data.get("per_station", {}),
            "v1_stations": v1_data.get("stations", []),
            "v2_stations": v2_data.get("stations", []),
            "v1_scanned_stations": self._discover_stations(self._get_patient_dir("v1", patient_id)),
            "v2_scanned_stations": self._discover_stations(self._get_patient_dir("v2", patient_id)),
            "overlay_images": overlay_images,
        }

    def get_available_organs_for_patient(self, patient_id: str) -> list[str]:
        """Check which organs actually have segmentation data for this patient."""
        patient_dir = os.path.join(self.vista3d_dir, str(patient_id))
        if not os.path.isdir(patient_dir):
            return []

        found_organs = set()
        for station in self._discover_stations(patient_dir):
            seg_path = os.path.join(patient_dir, station, "seg_trans.nii.gz")
            if not os.path.exists(seg_path):
                continue
            try:
                import nibabel as nib
                import numpy as np

                nii = nib.load(seg_path)
                seg_data = np.asanyarray(nii.dataobj)
                unique_labels = np.unique(seg_data)

                for label in unique_labels:
                    if label == 0:
                        continue
                    organ = get_organ_name(int(label))
                    if organ:
                        found_organs.add(organ)
            except Exception:
                continue

        return sorted(found_organs)


# ============================================================
# Tool Definition
# ============================================================

# Thread-local storage for concurrent agent execution
_thread_local = threading.local()

# Global loader instance (set by build_agent) — shared across threads (read-only after init)
_loader: Vista3DResultLoader = None

# Whether the LLM supports vision (inject overlay images into messages)
_use_vlm: bool = False

# Global API concurrency semaphore — gates all LLM calls to prevent API rate limiting
_api_semaphore = None


def set_max_api_concurrent(n: int):
    """Set the global API concurrency limit for LLM calls."""
    global _api_semaphore
    _api_semaphore = threading.Semaphore(n) if n and n > 0 else None


def _get_overlay_queue() -> dict:
    """Get thread-local overlay queue (each thread gets its own).

    Used for VLM injection — cleared after each custom_tools_node call.
    """
    if not hasattr(_thread_local, "overlay_queue"):
        _thread_local.overlay_queue = {}
    return _thread_local.overlay_queue


def _get_all_overlays() -> dict:
    """Get thread-local all-overlays dict (accumulates across all tool calls).

    Used for trajectory output — drained once by run_vista3d_agent at the end.
    Ensures overlay images are saved to tool_images even in VLM mode.
    """
    if not hasattr(_thread_local, "all_overlays"):
        _thread_local.all_overlays = {}
    return _thread_local.all_overlays


def _get_current_patient_id() -> str:
    """Get thread-local current patient ID."""
    return getattr(_thread_local, "patient_id", None)


def _set_current_patient_id(pid: str):
    """Set thread-local current patient ID."""
    _thread_local.patient_id = pid


@tool
def vista3d_segment(organ: str) -> str:
    """Get Vista3D segmentation volume for a specific organ, broken down per body station.

    Returns per-station volumes, slice ranges, and total volume. Use this first
    to understand the organ's spatial distribution across body stations.

    Args:
        organ: Organ name (e.g., "liver", "spleen", "pancreas", "kidney_left", "heart")

    Returns:
        Summary string with per-station volume breakdown.
    """
    if _loader is None:
        return "Error: Vista3D loader not initialized"

    patient_id = _get_current_patient_id()
    if patient_id is None:
        return "Error: No patient ID set"

    result = _loader.get_organ_volume(patient_id, organ)

    if not result["found"]:
        return f"{result['organ']}: Not found in segmentation data. {result.get('error', 'Organ may be outside scanned region.')}"

    per_st = result.get("per_station", {})
    lines = [
        f"{result['organ']}: total_volume={result['volume_ml']:.1f} ml (voxels={result['voxel_count']})",
        f"  Found in {len(per_st)} station(s):",
    ]
    # Iterate per_st directly (auto-discovered station names vary across
    # patients, e.g. dixon_28-48_W) — DEFAULT_STATIONS would hide them.
    for st in sorted(per_st, key=lambda s: int(s.split("_")[1])):
        d = per_st[st]
        region = STATION_REGIONS.get(st, "unknown")
        sr = d.get("slice_range")
        sr_str = f"slices {sr[0]}-{sr[1]}" if sr else "no slices"
        lines.append(
            f"  {st} ({region}): {d['volume_ml']:.1f} ml, "
            f"{d['voxel_count']} voxels, {d['num_slices']} slices "
            f"({sr_str}), max_slice={d['max_slice']} "
            f"(area={d['max_slice_area_mm2']:.1f} mm²)"
        )

    return "\n".join(lines)


@tool
def vista3d_compare_stations(organ: str) -> str:
    """Compare an organ's volume and morphology between dixon_8_W (thorax) and dixon_12_W (abdomen).

    Returns per-station volumes, slice overlap info, volume difference, and
    overlay images showing the organ highlighted on its largest slice in each station.
    Use this for organs that may span both stations (liver, pancreas, kidneys, spleen).

    Args:
        organ: Organ name (e.g., "liver", "pancreas", "kidney_left")

    Returns:
        Comparison summary with per-station volumes and overlay image references.
    """
    if _loader is None:
        return "Error: Vista3D loader not initialized"

    patient_id = _get_current_patient_id()
    if patient_id is None:
        return "Error: No patient ID set"

    cmp_data = _loader.compare_stations(patient_id, organ)

    if not cmp_data["found"]:
        return f"{cmp_data['organ']}: Not found in segmentation data."

    comparison = cmp_data["comparison"]
    overlays = cmp_data.get("overlay_images", {})

    lines = [
        f"{cmp_data['organ']} station comparison (dixon_8_W vs dixon_12_W):",
        f"  Total volume: {cmp_data['total_volume_ml']:.1f} ml",
        f"  Volume difference (d8 - d12): {cmp_data['volume_diff_ml']:.1f} ml",
        "",
    ]

    for st in ["dixon_8_W", "dixon_12_W"]:
        d = comparison.get(st, {})
        region = STATION_REGIONS.get(st, "unknown")
        if d.get("voxel_count", 0) > 0:
            sr = d.get("slice_range")
            sr_str = f"slices {sr[0]}-{sr[1]}" if sr else "none"
            lines.append(
                f"  {st} ({region}): {d['volume_ml']:.1f} ml, "
                f"{d['num_slices']} active slices ({sr_str}), "
                f"max_slice={d['max_slice']} (area={d['max_slice_area_mm2']:.1f} mm²)"
            )
        else:
            lines.append(f"  {st} ({region}): not found in this station")

    # Store overlay images for visual injection (not in text response)
    if overlays:
        _get_overlay_queue()[organ] = overlays
        _get_all_overlays()[organ] = overlays

    if overlays:
        lines.append("")
        lines.append(f"  [Overlay images generated for {len(overlays)} station(s) — see visual context]")

    return "\n".join(lines)


@tool
def vista3d_compare_visits(organ: str) -> str:
    """Compare an organ's volume between V1 (baseline) and V2 (follow-up) MRI visits.

    Returns V1 volume, V2 volume, volume change, per-station breakdown, and
    overlay images from both visits for visual comparison.
    This is the PRIMARY tool for analyzing organ volume changes over time.

    Args:
        organ: Organ name (e.g., "liver", "pancreas", "kidney_left", "spleen")

    Returns:
        Comparison summary with V1 vs V2 volumes and overlay image references.
    """
    if _loader is None:
        return "Error: Vista3D loader not initialized"

    patient_id = _get_current_patient_id()
    if patient_id is None:
        return "Error: No patient ID set"

    cmp_data = _loader.compare_visits(patient_id, organ)

    if not cmp_data["found"]:
        return f"{cmp_data['organ']}: Not found in segmentation data for either visit."

    v1_vol = cmp_data["v1_volume_ml"]
    v2_vol = cmp_data["v2_volume_ml"]
    v1_scan = cmp_data.get("v1_scanned_stations", [])
    v2_scan = cmp_data.get("v2_scanned_stations", [])

    def _visit_line(label, vol, found_stations, scanned):
        if vol > 0:
            return f"  {label}:  {vol:.1f} ml"
        # 0 ml → organ not in this visit's scanned region (don't say "0.0 ml",
        # which the model misreads as atrophy)
        return (f"  {label}:  NOT present in this visit's scan "
                f"(this visit scanned: {scanned})")

    lines = [
        f"{cmp_data['organ']} V1 vs V2 visit comparison:",
        _visit_line("V1 (baseline)", v1_vol, cmp_data.get("v1_stations", []), v1_scan),
        _visit_line("V2 (follow-up)", v2_vol, cmp_data.get("v2_stations", []), v2_scan),
    ]

    # Warn when the two visits scanned different body regions → volume
    # comparison is not meaningful.
    if v1_scan and v2_scan and set(v1_scan) != set(v2_scan):
        only_v1 = sorted(set(v1_scan) - set(v2_scan))
        only_v2 = sorted(set(v2_scan) - set(v1_scan))
        lines.append("  ⚠ V1 and V2 scanned DIFFERENT body regions "
                     f"(V1-only: {only_v1[:4]}, V2-only: {only_v2[:4]}) — "
                     "cross-visit volume comparison may be invalid.")
    elif v1_vol == 0 or v2_vol == 0:
        lines.append("  ⚠ Organ absent from one visit — volume change is not meaningful.")

    lines.append(f"  Volume change:  {cmp_data['volume_change_ml']:+.1f} ml ({cmp_data['volume_change_pct']:+.1f}%)")
    lines += ["", f"  V1 stations (organ found): {cmp_data.get('v1_stations', [])}",
              f"  V2 stations (organ found): {cmp_data.get('v2_stations', [])}", "",
              "  V1 per-station:"]

    for st, d in cmp_data.get("v1_per_station", {}).items():
        lines.append(f"    {st}: {d['volume_ml']:.1f} ml, {d['num_slices']} slices, max_slice={d['max_slice']}")

    lines.append("  V2 per-station:")
    for st, d in cmp_data.get("v2_per_station", {}).items():
        lines.append(f"    {st}: {d['volume_ml']:.1f} ml, {d['num_slices']} slices, max_slice={d['max_slice']}")

    # Store overlay images for visual injection / trajectory output
    overlays = cmp_data.get("overlay_images", {})
    if overlays and (overlays.get("v1") or overlays.get("v2")):
        _get_overlay_queue()[organ] = overlays
        _get_all_overlays()[organ] = overlays
        total = sum(len(v) for v in overlays.values())
        lines.append("")
        lines.append(f"  [Overlay images generated for {total} visit/station combo(s) — see visual context]")

    return "\n".join(lines)


# ============================================================
# LangGraph Nodes
# ============================================================

def create_llm_node(llm_with_tools):
    """Create the LLM node that decides tool calls or generates final answer."""

    def llm_node(state: Vista3DAgentState) -> dict:
        if _api_semaphore is not None:
            with _api_semaphore:
                response = llm_with_tools.invoke(state["messages"])
        else:
            response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return llm_node


def custom_tools_node(state: Vista3DAgentState) -> dict:
    """Execute tools and inject overlay images as HumanMessage for VLM analysis.

    Handles three tools: vista3d_segment, vista3d_compare_stations, vista3d_compare_visits.
    Overlay images from compare tools are injected as HumanMessage for VLM models,
    or stored in _overlay_queue for trajectory output in text mode.
    """
    from langchain_core.messages import ToolMessage

    last_message = state["messages"][-1]
    tool_messages = []
    visual_content_blocks = []

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"] or {}

        # Validate required "organ" argument
        if "organ" not in tool_args or not tool_args["organ"]:
            tool_messages.append(ToolMessage(
                content="Error: missing required 'organ' argument. Please provide an organ name (e.g., 'liver', 'spleen', 'pancreas').",
                tool_call_id=tc["id"],
            ))
            continue

        try:
            if tool_name == "vista3d_segment":
                result = vista3d_segment.invoke(tool_args)
            elif tool_name == "vista3d_compare_stations":
                result = vista3d_compare_stations.invoke(tool_args)
            elif tool_name == "vista3d_compare_visits":
                result = vista3d_compare_visits.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
        except Exception as e:
            result = f"Error executing {tool_name}: {type(e).__name__}: {str(e)[:200]}"

        tool_messages.append(ToolMessage(
            content=result,
            tool_call_id=tc["id"],
        ))

    # Consume overlay queue — inject as HumanMessage only for VLM models
    global _use_vlm
    overlay_queue = _get_overlay_queue()
    if overlay_queue and _use_vlm:
        for organ, overlays in overlay_queue.items():
            # Handle both formats:
            #   compare_stations: {station: b64}
            #   compare_visits: {v1: {station: b64}, v2: {station: b64}}
            for key, val in overlays.items():
                if isinstance(val, dict):
                    # Nested format (compare_visits): key="v1"/"v2", val={station: b64}
                    for station, b64 in val.items():
                        visual_content_blocks.append({
                            "type": "text",
                            "text": f"[{organ} overlay — {key.upper()} {station}]",
                        })
                        visual_content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        })
                else:
                    # Flat format (compare_stations): key=station, val=b64
                    visual_content_blocks.append({
                        "type": "text",
                        "text": f"[{organ} overlay — {key}]",
                    })
                    visual_content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{val}"},
                    })
        overlay_queue.clear()

        visual_msg = HumanMessage(content=visual_content_blocks)
        tool_messages.append(visual_msg)
    # When _use_vlm is False, overlays stay in _overlay_queue for trajectory output

    return {"messages": tool_messages}


def should_continue(state: Vista3DAgentState) -> str:
    """Route: if LLM wants to call tools → 'tools', else → END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def extract_answer_node(state: Vista3DAgentState) -> dict:
    """Extract answer from the final LLM response."""
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    import re
    # Extract answer letter
    match = re.search(r'Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)', content, re.IGNORECASE)
    answer = None
    if match:
        letters = sorted(set(l.strip().upper() for l in match.group(1).split(",") if l.strip()))
        if letters:
            answer = ",".join(letters)

    return {"answer": answer}


# ============================================================
# Graph Builder
# ============================================================

def build_vista3d_agent(
    vista3d_dir: str = DEFAULT_VISTA3D_DIR,
    vista3d_dir_v2: str = None,
    model: str = "qwen3.6-plus",
    api_key: str = None,
    base_url: str = None,
    use_vlm: bool = True,
):
    """Build the LangGraph Vista3D agent with V1/V2 visit comparison + VLM support.

    Args:
        vista3d_dir: Path to V1 (baseline) Vista3D segmentation results
        vista3d_dir_v2: Path to V2 (follow-up) Vista3D segmentation results
        model: LLM model name (qwen3.6-plus, qwen3.5-plus — both support vision)
        api_key: OpenAI-compatible API key
        base_url: API base URL
        use_vlm: Whether to inject overlay images into LLM messages (default True)

    Returns:
        Compiled LangGraph agent
    """
    global _loader, _use_vlm

    # Initialize loader with both visit directories
    _loader = Vista3DResultLoader(vista3d_dir, vista3d_dir_v2)

    # VLM flag — qwen3.6-plus and qwen3.5-plus both support vision input
    _use_vlm = use_vlm

    # Build LLM kwargs — disable thinking by default (SFT model trained without it).
    # Env overrides let non-qwen endpoints work: some hosts (e.g. idealab
    # claude-opus-4-8) REJECT temperature and/or the qwen-only enable_thinking
    # extra_body, and can be flaky (need more retries). Defaults are unchanged.
    llm_kwargs = {
        "model": model,
        "api_key": api_key or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "max_tokens": int(os.environ.get("MAX_RESPONSE_TOKENS", 2000)),
    }
    _temp = os.environ.get("LLM_TEMPERATURE", "0.0")
    if _temp.strip().lower() not in ("", "none", "null"):
        llm_kwargs["temperature"] = float(_temp)
    if os.environ.get("LLM_NO_EXTRA_BODY", "0") != "1":
        llm_kwargs["extra_body"] = {"enable_thinking": False}
    _retries = os.environ.get("LLM_MAX_RETRIES")
    if _retries:
        llm_kwargs["max_retries"] = int(_retries)

    llm = ChatOpenAI(**llm_kwargs)
    tools = [vista3d_segment, vista3d_compare_stations, vista3d_compare_visits]
    llm_with_tools = llm.bind_tools(tools)

    # Build graph
    graph = StateGraph(Vista3DAgentState)

    graph.add_node("llm", create_llm_node(llm_with_tools))
    graph.add_node("tools", custom_tools_node)
    graph.add_node("extract_answer", extract_answer_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: "extract_answer"})
    graph.add_edge("tools", "llm")
    graph.add_edge("extract_answer", END)

    return graph.compile()


# ============================================================
# Main Entry Point
# ============================================================

def run_vista3d_agent(
    question: str,
    options: dict,
    patient_id: str,
    patient_info: str = "",
    ground_truth: str = "",
    vista3d_dir: str = DEFAULT_VISTA3D_DIR,
    vista3d_dir_v2: str = None,
    model: str = "qwen3.6-plus",
    available_organs: list = None,
    include_preview_images: bool = True,
) -> dict:
    """Run the Vista3D agent for a single question.

    Args:
        question: Clinical question
        options: Answer options {A: "...", B: "...", ...}
        patient_id: Patient ID
        patient_info: Patient narrative
        ground_truth: Correct answer (for evaluation)
        vista3d_dir: Path to V1 (baseline) Vista3D data
        vista3d_dir_v2: Path to V2 (follow-up) Vista3D data
        model: LLM model name (qwen3.6-plus for text, qwen-vl-max for VLM)
        available_organs: Pre-computed list of available organs
        include_preview_images: Include d8/d12 preview images in initial message (VLM only)

    Returns:
        {
            "answer": "A",
            "correct": True/False,
            "messages": [...],  # full trajectory (standard OpenAI format)
            "tool_images": {},  # overlay images from tools (for VLM SFT augmentation)
        }
    """
    global _loader

    agent = build_vista3d_agent(vista3d_dir, vista3d_dir_v2, model, use_vlm=True)

    # Set patient context (thread-local for concurrent execution)
    _set_current_patient_id(str(patient_id))

    # Get available organs if not provided (for internal use, NOT exposed to LLM)
    if available_organs is None:
        available_organs = _loader.get_available_organs_for_patient(str(patient_id))

    opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    # Build initial message — optionally include preview images from key stations
    prompt_text = (
        f"Patient ID: {patient_id}\n"
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_text}\n\n"
        f"Available organs in this patient's segmentation: {', '.join(available_organs[:20])}\n\n"
        f"Instructions:\n"
        f"1. Identify the 1-3 organs most relevant to the question (1 if the question targets a single organ, up to 3 if it involves multiple organs or is general).\n"
        f"2. Call vista3d_compare_visits ONLY for those organs — do NOT call tools for other organs.\n"
        f"3. Analyze the volume change data AND overlay images, then write your 5-phase FINAL response and STOP.\n"
        f"   Do NOT call more than 3 tools total. Do NOT call the same tool twice for the same organ."
    )

    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]

    # Pre-include preview images from d8 and d12 for VLM visual context
    if include_preview_images and _use_vlm:
        content_blocks = [{"type": "text", "text": prompt_text}]
        for st in ["dixon_8_W", "dixon_12_W"]:
            preview_path = os.path.join(vista3d_dir, str(patient_id), st, "preview_slices.png")
            if os.path.exists(preview_path):
                with open(preview_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_blocks.append({
                    "type": "text",
                    "text": f"[Preview: {st} ({STATION_REGIONS.get(st, '?')}) — 3 representative axial slices with full segmentation overlay]",
                })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        initial_messages.append(HumanMessage(content=content_blocks))
    else:
        initial_messages.append(HumanMessage(content=prompt_text))

    initial_state = Vista3DAgentState(
        messages=initial_messages,
        patient_id=str(patient_id),
        patient_info=patient_info,
        question=question,
        options=options,
        answer=None,
        correct=None,
    )

    result = agent.invoke(initial_state, {"recursion_limit": 50})

    # Drain all overlays (accumulated across all tool calls, for trajectory output)
    all_overlays = _get_all_overlays()
    tool_images = dict(all_overlays) if all_overlays else {}
    all_overlays.clear()
    # Also clear injection queue (in case text mode left overlays there)
    _get_overlay_queue().clear()

    # Check correctness
    answer = result.get("answer", "")
    correct = None
    if ground_truth and answer:
        norm = lambda a: ",".join(sorted(set(l.strip().upper() for l in str(a).split(",") if l.strip())))
        correct = (norm(answer) == norm(ground_truth))

    return {
        "answer": answer,
        "correct": correct,
        "messages": result["messages"],
        "tool_images": tool_images,
    }


# ============================================================
# Batch Runner
# ============================================================

def run_batch(
    parquet_path: str,
    vista3d_dir: str = DEFAULT_VISTA3D_DIR,
    vista3d_dir_v2: str = None,
    output_dir: str = "/tmp/vista3d_agent_output",
    max_patients: int = None,
    model: str = "qwen3.6-plus",
):
    """Run Vista3D agent on multiple patients from parquet.

    Args:
        parquet_path: Path to patient data parquet
        vista3d_dir: Path to V1 (baseline) Vista3D segmentation data
        vista3d_dir_v2: Path to V2 (follow-up) Vista3D segmentation data
        output_dir: Output directory for trajectories
        max_patients: Limit number of patients (for testing)
        model: LLM model name
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from distill.tool.common.test_medpanel import load_patient_data, parse_question

    os.makedirs(output_dir, exist_ok=True)

    # Get patient IDs from Vista3D data
    if not os.path.isdir(vista3d_dir):
        print(f"Error: Vista3D directory not found: {vista3d_dir}")
        return

    patient_ids = sorted([d for d in os.listdir(vista3d_dir) if d.isdigit()])
    if max_patients:
        patient_ids = patient_ids[:max_patients]

    print(f"Found {len(patient_ids)} patients in Vista3D data")

    results = []
    for i, pid in enumerate(patient_ids):
        print(f"\n[{i+1}/{len(patient_ids)}] Patient {pid}...")

        try:
            data = load_patient_data(parquet_path, pid)
        except Exception as e:
            print(f"  [Skip] Patient {pid} not in parquet: {e}")
            continue

        patient_info = data.get("patient_info", "")
        problems = data.get("problems", [])
        solutions = data.get("solutions", [])

        for qidx, problem in enumerate(problems):
            question, options = parse_question(problem)
            if not question:
                continue

            gt = solutions[qidx] if qidx < len(solutions) else ""

            result = run_vista3d_agent(
                question=question,
                options=options,
                patient_id=pid,
                patient_info=patient_info,
                ground_truth=gt,
                vista3d_dir=vista3d_dir,
                vista3d_dir_v2=vista3d_dir_v2,
                model=model,
            )

            result["patient_id"] = pid
            result["question_idx"] = qidx
            result["ground_truth"] = gt
            results.append(result)

            status = "✓" if result.get("correct") else "✗"
            print(f"  Q{qidx}: {result.get('answer', '?')} (GT: {gt}) {status}")

    # Save results
    output_path = os.path.join(output_dir, "results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            # Convert messages to serializable format
            msgs = []
            for m in r.get("messages", []):
                msg_dict = {
                    "role": getattr(m, "type", str(type(m))),
                    "content": getattr(m, "content", ""),
                }
                if hasattr(m, "tool_calls") and m.tool_calls:
                    msg_dict["tool_calls"] = m.tool_calls
                if hasattr(m, "tool_call_id"):
                    msg_dict["tool_call_id"] = m.tool_call_id
                msgs.append(msg_dict)

            entry = {
                "patient_id": r.get("patient_id"),
                "question_idx": r.get("question_idx"),
                "answer": r.get("answer"),
                "ground_truth": r.get("ground_truth"),
                "correct": r.get("correct"),
                "messages": msgs,
                "tool_images": r.get("tool_images", {}),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    print(f"\n{'='*60}")
    print(f"Vista3D Agent Batch Results")
    print(f"{'='*60}")
    print(f"  Patients: {len(patient_ids)}")
    print(f"  Questions: {total}")
    print(f"  Correct: {correct}")
    if total > 0:
        print(f"  Accuracy: {correct/total*100:.1f}%")
    print(f"  Results: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vista3D LangGraph Agent")
    parser.add_argument("--vista3d-dir", type=str, default=DEFAULT_VISTA3D_DIR)
    parser.add_argument("--vista3d-dir-v2", type=str, default=None,
                        help="Path to V2 (follow-up) Vista3D data")
    parser.add_argument("--parquet", type=str, default="/tmp/mys/dataset/opd_rl_all_0.9/train_dir/train.parquet")
    parser.add_argument("--output", type=str, default="/tmp/vista3d_agent_output")
    parser.add_argument("--max-patients", type=int, default=2)
    parser.add_argument("--model", type=str, default="qwen3.6-plus")
    args = parser.parse_args()

    run_batch(
        parquet_path=args.parquet,
        vista3d_dir=args.vista3d_dir,
        vista3d_dir_v2=args.vista3d_dir_v2,
        output_dir=args.output,
        max_patients=args.max_patients,
        model=args.model,
    )
