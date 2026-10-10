"""
MedPanel v2 — LangGraph-based autonomous medical imaging agent.

The model autonomously decides which modalities to examine, which slices
to view, and compares Visit 1 vs Visit 2 to answer clinical questions.

Tools:
  - list_available_images: List all available imaging modalities for the patient
  - view_image: View a specific slice from a modality at a given visit
  - compare_visits: Side-by-side V1 vs V2 comparison of the same slice
"""
import base64
import io
import os
import re
import threading
from typing import Annotated, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_MRI_DIR = "/mnt/faster_nas/benyou.mys/dataset/ukb_mri_all"
DEFAULT_VISTA3D_DIR = "/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d"
DEFAULT_VISTA3D_DIR_V2 = "/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2"

MAX_TOOL_CALLS = 4
IMAGE_SIZE = 512

MODALITY_DESCRIPTIONS = {
    "brain_t1": "Brain T1 MRI — anatomical brain structure",
    "brain_t2_flair": "Brain T2 FLAIR MRI — white matter lesion pathology",
    "heart_long_axis": "Heart Long-axis CineMRI — 2CH/4CH cardiac views",
    "heart_short_axis": "Heart Short-axis CineMRI — cross-sectional cardiac slices",
    "liver_ideal": "Liver IDEAL MRI — fat/water separation for steatosis",
    "pancreas_ge": "Pancreas Gradient-Echo MRI — pancreatic structure",
}

DIXON_STATIONS = ["dixon_4_W", "dixon_8_W", "dixon_12_W", "dixon_16_W", "dixon_20_W", "dixon_24_W"]
DIXON_REGIONS = {
    "dixon_4_W": "head_neck", "dixon_8_W": "thorax_upper",
    "dixon_12_W": "thorax_lower_abdomen", "dixon_16_W": "abdomen",
    "dixon_20_W": "pelvis", "dixon_24_W": "legs",
}

SYSTEM_PROMPT = """You are MedPanel, an autonomous medical imaging analysis agent. You examine longitudinal medical imaging data (Visit 1 vs Visit 2) to answer clinical questions about disease progression, changes, and diagnosis.

## Available Tools
You have tools to examine patient imaging data:
1. **list_available_images** — List all available imaging modalities and their metadata
2. **view_image** — View a specific axial slice from a modality at a given visit (returns an image)
3. **compare_visits** — Side-by-side comparison of the same slice between Visit 1 and Visit 2

## Strategy
1. Call list_available_images first to see what's available
2. Select 1-2 most relevant modalities based on the clinical question
3. Use view_image or compare_visits to examine specific slices
4. Limit yourself to {max_calls} tool calls total — be efficient!
5. After examining images, provide your final diagnosis

## Rules
- Be concise in tool usage — don't examine more than 2 modalities
- Pick the most informative slice (usually the middle slice or where pathology is visible)
- Never call the same tool with the same arguments twice
- When you have enough information, write your final response

## Final Response Format
Your final response MUST contain ALL 5 phases in this exact format:

## Phase 1: Modality Selection
- Selected: [modalities chosen]
- Reasoning: [why these modalities are relevant]

## Phase 2: Visit 1 Analysis
- Findings: [key observations from baseline images]

## Phase 3: Visit 2 Analysis & Changes
- Findings: [key observations from follow-up images]
- Interval Changes: [what changed between visits]

## Phase 4: Insight Synthesis
- Clinical Insight: [Synthesize ALL prior phases into ONE coherent clinical insight. Explicitly combine: Phase 1 (why these modalities were chosen), Phase 2 (Visit 1 findings), Phase 3 (Visit 2 findings and interval changes), and the imaging evidence you obtained from your tool calls. State clearly what the evidence shows and what it implies for the question. Base this ONLY on evidence you actually observed — do not speculate beyond the data.]

## Phase 5: Final Diagnosis
- Answer: [single letter A/B/C/D]
- Reasoning: [MUST be derived directly from your Phase 4 Clinical Insight. The answer must follow logically from Phase 4 and be consistent with it. If Phase 4 concludes there is no evidence supporting any of the listed options, select "None of the above" if it is an option.]
"""

# ── Thread-local state ─────────────────────────────────────────────────────

_thread_local = threading.local()
_api_semaphore = None


def set_max_api_concurrent(n):
    global _api_semaphore
    _api_semaphore = threading.Semaphore(n)


def _get_patient_id():
    return getattr(_thread_local, "patient_id", None)


def _set_patient_id(pid):
    _thread_local.patient_id = pid


def _get_image_queue():
    if not hasattr(_thread_local, "image_queue"):
        _thread_local.image_queue = []
    return _thread_local.image_queue


def _get_all_images():
    if not hasattr(_thread_local, "all_images"):
        _thread_local.all_images = []
    return _thread_local.all_images


def _get_mri_dir():
    return getattr(_thread_local, "mri_dir", DEFAULT_MRI_DIR)


def _get_vista3d_dir():
    return getattr(_thread_local, "vista3d_dir", DEFAULT_VISTA3D_DIR)


def _get_vista3d_dir_v2():
    return getattr(_thread_local, "vista3d_dir_v2", DEFAULT_VISTA3D_DIR_V2)


# ── Image loading ──────────────────────────────────────────────────────────

_volume_cache = {}
_cache_lock = threading.Lock()
_MAX_CACHE = 20


def _load_volume(patient_id, modality, visit):
    """Load a NIfTI volume. visit: 'v1' or 'v2'."""
    cache_key = (patient_id, modality, visit)
    with _cache_lock:
        if cache_key in _volume_cache:
            return _volume_cache[cache_key]

    # Determine directory
    if modality.startswith("dixon_"):
        # Whole Body Dixon — from vista3d dirs
        if visit == "v1":
            base_dir = _get_vista3d_dir()
            patient_dir = os.path.join(base_dir, patient_id)
        else:
            base_dir = _get_vista3d_dir_v2()
            patient_dir = os.path.join(base_dir, f"{patient_id}_visit3")
            if not os.path.isdir(patient_dir):
                patient_dir = os.path.join(base_dir, patient_id)
        nifti_path = os.path.join(patient_dir, modality, "input.nii.gz")
    else:
        # Other modalities — from ukb_mri_all
        visit_num = "2" if visit == "v1" else "3"
        patient_dir = os.path.join(_get_mri_dir(), f"{patient_id}_visit{visit_num}")
        mod_dir = os.path.join(patient_dir, modality)
        nifti_path = os.path.join(mod_dir, "input.nii.gz")
        if not os.path.exists(nifti_path):
            for fname in ["T1_orig_defaced.nii.gz", "T2_FLAIR_orig_defaced.nii.gz"]:
                alt = os.path.join(mod_dir, fname)
                if os.path.exists(alt):
                    nifti_path = alt
                    break

    if not os.path.exists(nifti_path):
        return None

    try:
        nii = nib.load(nifti_path)
        data = np.asanyarray(nii.dataobj)
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
        result = (data, spacing)
        with _cache_lock:
            if len(_volume_cache) >= _MAX_CACHE:
                _volume_cache.clear()
            _volume_cache[cache_key] = result
        return result
    except Exception:
        return None


def _render_slice(data, slice_idx, axis=2):
    """Render a slice as base64 PNG. Returns (b64_str, actual_slice_idx)."""
    n_slices = data.shape[axis]
    slice_idx = max(0, min(slice_idx, n_slices - 1))
    if axis == 0:
        slc = data[slice_idx, :, :]
    elif axis == 1:
        slc = data[:, slice_idx, :]
    else:
        slc = data[:, :, slice_idx]

    slc = np.rot90(slc)
    vmin, vmax = np.percentile(slc, [1, 99])
    slc = np.clip((slc - vmin) / (vmax - vmin + 1e-8), 0, 1)
    slc = (slc * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(slc, mode="L").save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return b64, slice_idx


def _render_comparison(data_v1, data_v2, slice_idx, axis=2):
    """Render side-by-side V1 vs V2 comparison. Returns (b64_str, slice_idx)."""
    n_slices = max(data_v1.shape[axis], data_v2.shape[axis])
    slice_idx = max(0, min(slice_idx, n_slices - 1))

    def get_slice(data, idx, ax):
        idx = max(0, min(idx, data.shape[ax] - 1))
        if ax == 0:
            return data[idx, :, :]
        elif ax == 1:
            return data[:, idx, :]
        else:
            return data[:, :, idx]

    s1 = np.rot90(get_slice(data_v1, slice_idx, axis))
    s2 = np.rot90(get_slice(data_v2, slice_idx, axis))

    combined = np.hstack([s1, s2])
    vmin, vmax = np.percentile(combined, [1, 99])
    s1 = np.clip((s1 - vmin) / (vmax - vmin + 1e-8), 0, 1)
    s2 = np.clip((s2 - vmin) / (vmax - vmin + 1e-8), 0, 1)

    combined = np.hstack([s1, s2])
    combined = (combined * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(combined, mode="L").save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return b64, slice_idx


# ── Tool definitions ───────────────────────────────────────────────────────

@tool
def list_available_images() -> str:
    """List all available imaging modalities for this patient.
    Returns modality names, shapes, and descriptions."""
    patient_id = _get_patient_id()
    if not patient_id:
        return "Error: no patient context"

    available = []
    visit_nums = {"v1": "2", "v2": "3"}

    for mod_name, desc in MODALITY_DESCRIPTIONS.items():
        for visit_key, visit_num in visit_nums.items():
            result = _load_volume(patient_id, mod_name, visit_key)
            if result is not None:
                data, spacing = result
                available.append(
                    f"  {mod_name} ({visit_key}): shape={data.shape}, "
                    f"spacing={tuple(round(s, 2) for s in spacing)} — {desc}"
                )

    for station in DIXON_STATIONS:
        for visit_key in ["v1", "v2"]:
            result = _load_volume(patient_id, station, visit_key)
            if result is not None:
                data, spacing = result
                region = DIXON_REGIONS.get(station, "unknown")
                available.append(
                    f"  {station} ({visit_key}): shape={data.shape}, "
                    f"spacing={tuple(round(s, 2) for s in spacing)} — "
                    f"Whole-body Dixon ({region})"
                )

    if not available:
        return "No imaging data available for this patient."
    return "Available imaging modalities:\n" + "\n".join(available)


@tool
def view_image(modality: str, visit: str, slice_idx: int = -1) -> str:
    """View a specific axial slice from a modality.
    Args:
        modality: Modality name (e.g. 'brain_t1', 'heart_short_axis', 'liver_ideal', 'dixon_8_W')
        visit: 'v1' for baseline or 'v2' for follow-up
        slice_idx: Slice index (use -1 for middle slice)
    Returns the image and metadata."""
    patient_id = _get_patient_id()
    if not patient_id:
        return "Error: no patient context"

    result = _load_volume(patient_id, modality, visit)
    if result is None:
        return f"Error: modality '{modality}' not available for {visit}"

    data, spacing = result
    n_slices = data.shape[2]
    if slice_idx < 0 or slice_idx >= n_slices:
        slice_idx = n_slices // 2

    b64, actual_idx = _render_slice(data, slice_idx)
    _get_image_queue().append({"modality": modality, "visit": visit, "slice": actual_idx, "b64": b64})
    _get_all_images().append({"modality": modality, "visit": visit, "slice": actual_idx, "b64": b64})

    return (f"Viewing {modality} {visit}, slice {actual_idx}/{n_slices - 1}. "
            f"Shape: {data.shape}, Spacing: {tuple(round(s, 2) for s in spacing)} mm. "
            f"Image provided.")


@tool
def compare_visits(modality: str, slice_idx: int = -1) -> str:
    """Compare the same slice between Visit 1 and Visit 2 side-by-side.
    Args:
        modality: Modality name (e.g. 'brain_t1', 'liver_ideal', 'dixon_8_W')
        slice_idx: Slice index (use -1 for middle slice)
    Returns a side-by-side comparison image."""
    patient_id = _get_patient_id()
    if not patient_id:
        return "Error: no patient context"

    v1 = _load_volume(patient_id, modality, "v1")
    v2 = _load_volume(patient_id, modality, "v2")
    if v1 is None or v2 is None:
        return f"Error: {modality} not available for both visits"

    data_v1, spacing_v1 = v1
    data_v2, spacing_v2 = v2
    n_slices = max(data_v1.shape[2], data_v2.shape[2])
    if slice_idx < 0 or slice_idx >= n_slices:
        slice_idx = n_slices // 2

    b64, actual_idx = _render_comparison(data_v1, data_v2, slice_idx)
    _get_image_queue().append({
        "modality": modality, "visit": "comparison",
        "slice": actual_idx, "b64": b64, "type": "comparison"
    })
    _get_all_images().append({
        "modality": modality, "visit": "comparison",
        "slice": actual_idx, "b64": b64, "type": "comparison"
    })

    return (f"Side-by-side comparison of {modality}, slice {actual_idx}. "
            f"V1 shape: {data_v1.shape}, V2 shape: {data_v2.shape}. "
            f"Image provided.")


# ── LangGraph state and nodes ──────────────────────────────────────────────

class MedPanelAgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    patient_id: str
    patient_info: str
    question: str
    options: dict
    answer: Optional[str]
    correct: Optional[bool]


def _build_llm_node(llm_with_tools):
    def llm_node(state):
        if _api_semaphore:
            with _api_semaphore:
                response = llm_with_tools.invoke(state["messages"])
        else:
            response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    return llm_node


def _custom_tools_node(state):
    last_message = state["messages"][-1]
    tool_messages = []
    for tc in last_message.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "list_available_images":
            result = list_available_images.invoke(args)
        elif name == "view_image":
            result = view_image.invoke(args)
        elif name == "compare_visits":
            result = compare_visits.invoke(args)
        else:
            result = f"Unknown tool: {name}"

        tool_messages.append(ToolMessage(content=result, tool_call_id=tc_id, name=name))

    # Inject images as HumanMessage if any
    image_queue = _get_image_queue()
    if image_queue:
        content_blocks = []
        for img in image_queue:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img['b64']}"}
            })
        image_queue.clear()
        if content_blocks:
            tool_messages.append(HumanMessage(content=content_blocks))

    return {"messages": tool_messages}


def _should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def _extract_answer_node(state):
    last_ai = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            last_ai = msg
            break
    if last_ai is None:
        return {"answer": "A"}

    content = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
    m = re.search(r"Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)", content, re.IGNORECASE)
    if m:
        letters = sorted(set(l.strip().upper() for l in m.group(1).split(",") if l.strip()))
        return {"answer": ",".join(letters) if letters else "A"}
    return {"answer": "A"}


def build_medpanel_v2_agent(model, api_key, base_url, max_tokens=2000):
    """Build the LangGraph agent."""
    llm_kwargs = {
        "model": model,
        "api_key": api_key or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("MAX_RESPONSE_TOKENS", max_tokens)),
    }
    llm_kwargs["extra_body"] = {"enable_thinking": False}

    llm = ChatOpenAI(**llm_kwargs)
    tools = [list_available_images, view_image, compare_visits]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(MedPanelAgentState)
    graph.add_node("llm", _build_llm_node(llm_with_tools))
    graph.add_node("tools", _custom_tools_node)
    graph.add_node("extract_answer", _extract_answer_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", _should_continue, {"tools": "tools", END: "extract_answer"})
    graph.add_edge("tools", "llm")
    graph.add_edge("extract_answer", END)

    return graph.compile()


def run_medpanel_v2_agent(
    question, options, patient_id, patient_info, ground_truth,
    model="qwen3.7-plus",
    api_key=None, base_url=None,
    mri_dir=DEFAULT_MRI_DIR,
    vista3d_dir=DEFAULT_VISTA3D_DIR,
    vista3d_dir_v2=DEFAULT_VISTA3D_DIR_V2,
    max_tokens=2000,
):
    """Run the MedPanel v2 agent for a single question."""
    # Reset thread-local state
    _set_patient_id(patient_id)
    _thread_local.image_queue = []
    _thread_local.all_images = []
    _thread_local.mri_dir = mri_dir
    _thread_local.vista3d_dir = vista3d_dir
    _thread_local.vista3d_dir_v2 = vista3d_dir_v2

    agent = build_medpanel_v2_agent(model, api_key, base_url, max_tokens)

    system_prompt = SYSTEM_PROMPT.format(max_calls=MAX_TOOL_CALLS)
    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())

    user_content = (
        f"Patient ID: {patient_id}\n\n"
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Examine the available imaging data to answer this question. "
        f"Use at most {MAX_TOOL_CALLS} tool calls. "
        f"Be efficient — select the most relevant modality and examine key slices."
    )

    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    try:
        result = agent.invoke(
            {"messages": initial_messages, "patient_id": patient_id, "patient_info": patient_info,
             "question": question, "options": options, "answer": None, "correct": None},
            {"recursion_limit": 30},
        )
    except Exception as e:
        return {"answer": "A", "correct": ground_truth == "A", "messages": initial_messages,
                "tool_images": [], "error": str(e)}

    answer = result.get("answer", "A")
    if ground_truth:
        norm = lambda a: ",".join(sorted(set(l.strip().upper() for l in str(a).split(",") if l.strip())))
        correct = norm(answer) == norm(ground_truth)
    else:
        correct = None
    tool_images = list(_get_all_images())

    return {
        "answer": answer,
        "correct": correct,
        "messages": result["messages"],
        "tool_images": tool_images,
    }
