#!/usr/bin/env python3
"""medboard_tool.py — integrated multi-source review agent (recording side).

The medboard agent reviews a patient case by combining:
- medpanel_v2 imaging tools (list/view/compare slices)
- vista3d volumetry tools (segment/compare stations/compare visits)
- MedGemma report tools (list_available_reports / get_report: findings,
  impression, followup) — reports are treated as CLAIMS TO VERIFY
- skill memory (recall_skills before analysis, write_skill after solving)

For trajectory recording all tools are CACHE-BACKED: they replay the same
precomputed results the RL mock tools use (medpanel/vista3d tool caches +
the medgemma report cache), so recordings are deterministic, GPU-free, and
directly reusable for RL.

Graph: START -> llm -> [tools | extract_answer] , same as medpanel_v2/vista3d.
"""
import base64
import json
import os
import re
import threading
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

MAX_TOOL_CALLS = 12

# Cache locations. Overridable with environment variables; the run/*.sh
# pipeline exports the DATA_ROOT-based defaults.
_MEDPANEL_CACHE_DIR = os.environ.get(
    "MEDPANEL_V2_TOOL_CACHE_DIR", "/mnt/faster_nas/benyou.mys/medixrl/data/medpanel_v2_tool_cache")
_VISTA3D_CACHE_DIR = os.environ.get(
    "VISTA3D_TOOL_CACHE_DIR", "/mnt/faster_nas/benyou.mys/medixrl/data/vista3d_tool_cache")
DEFAULT_MEDPANEL_CACHE = os.path.join(_MEDPANEL_CACHE_DIR, "tool_cache.json")
DEFAULT_VISTA3D_CACHE = os.path.join(_VISTA3D_CACHE_DIR, "tool_cache.json")
DEFAULT_REPORT_CACHE = os.environ.get(
    "MEDGEMMA_REPORT_CACHE", "/mnt/faster_nas/benyou.mys/medixrl/data/medgemma_report_cache/tool_cache.json")
DEFAULT_SKILL_STORE = os.environ.get(
    "MEDBOARD_SKILL_STORE", "/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store/skill_store.jsonl")
MEDGEMMA_IMAGE_BASE = os.environ.get(
    "MEDGEMMA_IMAGE_BASE", "/mnt/faster_nas/benyou.mys/dataset/medgemma_0820")
MEDGEMMA_MANIFEST = os.path.join(MEDGEMMA_IMAGE_BASE, "manifest.json")

SYSTEM_PROMPT = """You are MedBoard, the FINAL ARBITRATION agent. Single-source evaluators on this task average only ~60-70% accuracy; your job is to integrate ALL evidence sources and beat any single one. Tool outputs vary in value: some sharpen your judgment, some are noise or even misleading — you are the gatekeeper who decides what to trust.

## Available evidence sources
1. Imaging: list_available_images, view_image, compare_visits (MRI slices)
2. Volumetry: vista3d_segment, vista3d_compare_stations, vista3d_compare_visits (organ volume quantification)
3. MedGemma reports: list_available_reports, get_report(organ, report_type) with report_type in findings|impression|followup. These are DRAFT reports from an automated system — they may contain hallucinations. Treat them as claims to verify, never as ground truth.
4. Skill library: recall_skills(organ, keywords) retrieves distilled experience from similar past cases (including lessons from past failures).

## Required workflow
Work through these steps with tool calls, taking brief notes as you go:
1. Call recall_skills for the implicated organ(s) FIRST.
2. Review evidence from multiple sources (images, volumes, reports) — use each source for what it is best at.
3. Actively DOUBT: mark key claims [VERIFIED], [SUSPICIOUS] or [CONTRADICTED]; verify every decision-relevant suspicious claim with a direct tool call.
4. Reflect: build the causal chain and explicitly consider what could mislead you here.

When done calling tools, your FINAL message (no tool calls) MUST contain ALL five headers below, in order, under 2000 characters total:

## Phase 1: Skill Recall & Review Plan
Which skills were recalled; which sources you will consult and why.
## Phase 2: Multi-source Review
Concrete findings from imaging/volumetry/reports, with numbers.
## Phase 3: Doubt & Verification
Key claims with [VERIFIED]/[SUSPICIOUS]/[CONTRADICTED] labels and what targeted checks showed.
## Phase 4: Reflective Insight
Clinical Insight: <the causal chain from verified evidence toward the answer, PLUS what could have misled you and why you rejected it>
## Phase 5: Final Answer
Answer: <letter(s)>

Rules:
- Maximum {max_calls} tool calls total (hard-enforced). Once you have used about 9 calls, STOP calling tools and write your final report with the evidence you have. Prefer high-value calls: recall_skills first, vista3d_compare_visits for change questions, get_report for report evidence, view_image only for targeted verification. Skip sources that are genuinely irrelevant to THIS question.
- Never copy a report conclusion without verification when a quantitative tool can check it.
- The Answer: line in Phase 5 is mandatory; your answer is extracted ONLY from this message.
- Do NOT call write_skill in this stage; skill recording happens in a later review stage."""

REFLECTION_PROMPT_CORRECT = """Your answer ({answer}) is CORRECT.

Now write Phase 6: Success Summary. In a few sentences, state why your evidence path worked. Then call write_skill ONCE with a concise reusable skill for similar future patients (patient_type, guidance, organ, keywords). Keep the skill short and actionable."""

REFLECTION_PROMPT_WRONG = """Your answer was {answer}, but the CORRECT answer is {gt}.

Now write Phase 6: Failure Reflection:
1. Root cause: which evidence was misread, which [SUSPICIOUS] report claim was accepted without verification, or which tool should have been called but wasn't?
2. Correction: how should a similar case be handled next time?
Then call write_skill ONCE recording this lesson (patient_type, guidance, pitfalls, organ, keywords) so the same mistake is avoided on similar future patients."""


class MedBoardAgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    patient_id: str
    question: str
    options: dict
    answer: Optional[str]


# ── Cache + thread-local context ───────────────────────────────────────────

_thread_local = threading.local()
_cache_lock = threading.Lock()
_caches = {}

# Module-level cache PATHS (thread-visible). init_medboard_caches writes these;
# _get_cache/_load_skill_store read them. Must NOT be thread-local, else
# ThreadPoolExecutor workers fall back to defaults and miss the passed caches.
_CACHE_PATHS = {
    "medpanel": DEFAULT_MEDPANEL_CACHE,
    "vista3d": DEFAULT_VISTA3D_CACHE,
    "report": DEFAULT_REPORT_CACHE,
    "skill_store": DEFAULT_SKILL_STORE,
}


def _load_json(path):
    with _cache_lock:
        if path not in _caches:
            with open(path) as f:
                _caches[path] = json.load(f)
        return _caches[path]


def _load_skill_store(path):
    with _cache_lock:
        key = path + "::jsonl"
        if key not in _caches:
            skills = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        skills.append(json.loads(line))
            _caches[key] = skills
        return _caches[key]


def init_medboard_caches(medpanel_cache=DEFAULT_MEDPANEL_CACHE,
                         vista3d_cache=DEFAULT_VISTA3D_CACHE,
                         report_cache=DEFAULT_REPORT_CACHE,
                         skill_store=DEFAULT_SKILL_STORE):
    """Preload caches (call once before batch recording).

    Paths are stored module-level (not thread-local) so ThreadPoolExecutor
    worker threads see them; thread-locals set in the main thread are
    invisible to workers and previously caused 'No MedGemma reports'.
    """
    _CACHE_PATHS["medpanel"] = medpanel_cache
    _CACHE_PATHS["vista3d"] = vista3d_cache
    _CACHE_PATHS["report"] = report_cache
    _CACHE_PATHS["skill_store"] = skill_store
    _load_json(medpanel_cache)
    _load_json(vista3d_cache)
    _load_json(report_cache)
    _load_skill_store(skill_store)


def _ctx():
    return _thread_local


def _get_cache(which):
    return _load_json(_CACHE_PATHS[which])


def _pid():
    return getattr(_thread_local, "patient_id", "")


def _queue_images(entry, cache_dir, meta_extra=None):
    """Load cache entry PNGs -> base64, push to image queue + all_images."""
    for e in (entry.get("images") or [])[:2]:
        p = os.path.join(cache_dir, e["path"])
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        meta = {"b64": b64, "label": e.get("label", ""), "src_path": e["path"]}
        if meta_extra:
            meta.update(meta_extra)
        _thread_local.image_queue.append(meta)
        _thread_local.all_images.append(meta)


def _resolve_organ(organ: str) -> str:
    organ = (organ or "").strip().lower()
    try:
        from distill.tool.anatomist.anatomist_label_map import resolve_organ
        return resolve_organ(organ) or organ
    except Exception:
        return organ


def _queue_medgemma_images(organ: str):
    """Queue the MedGemma representative images (montage/overview preferred) for
    an organ from the NAS manifest, so the model sees what the report used.

    Images are resized (longest edge 512, aspect kept) before queueing: the
    montages are large grids (up to ~2100px) whose raw size costs up to ~4200
    vLLM vision tokens each, which would blow the RL token budget."""
    try:
        man = _load_json(MEDGEMMA_MANIFEST)
    except Exception:
        return
    paths = []
    for visit in ("visit_2", "visit_3"):
        paths += man.get(f"{_pid()}::{organ}::{visit}", [])
    if not paths:
        return
    paths.sort(key=lambda p: 0 if ("montage" in p or "overview" in p) else 1)
    existing = {m.get("src_path") for m in _thread_local.all_images}
    new = [p for p in paths if p not in existing]
    if not new:
        return
    import base64
    import io
    from PIL import Image
    for p in new[:2]:
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
            b64 = base64.b64encode(buf.getvalue()).decode()
            meta = {"b64": b64, "label": "medgemma " + organ, "src_path": p}
            _thread_local.image_queue.append(meta)
            _thread_local.all_images.append(meta)
        except Exception:
            continue


# ── Tools (cache-backed) ───────────────────────────────────────────────────

@tool
def list_available_images() -> str:
    """List all available imaging modalities for this patient."""
    entry = _get_cache("medpanel").get(f"{_pid()}::list_available_images")
    return entry["text"] if entry else "No imaging data available for this patient."


@tool
def view_image(modality: str, visit: str, slice_idx: int = -1) -> str:
    """View a specific axial slice from a modality at a given visit ('v1' or 'v2')."""
    entry = _get_cache("medpanel").get(f"{_pid()}::view_image::{modality}::{visit}")
    if entry is None:
        return f"Error: modality '{modality}' not available for {visit}"
    _queue_images(entry, _MEDPANEL_CACHE_DIR,
                  {"modality": modality, "visit": visit, "type": "single"})
    return entry["text"]


@tool
def compare_visits(modality: str, slice_idx: int = -1) -> str:
    """Compare the same slice between Visit 1 and Visit 2 side-by-side."""
    entry = _get_cache("medpanel").get(f"{_pid()}::compare_visits::{modality}")
    if entry is None:
        return f"Error: {modality} not available for both visits"
    _queue_images(entry, _MEDPANEL_CACHE_DIR,
                  {"modality": modality, "type": "comparison"})
    return entry["text"]


@tool
def vista3d_segment(organ: str) -> str:
    """Get Vista3D segmentation volume breakdown for an organ, per body station."""
    key = f"{_pid()}::vista3d_segment::{_resolve_organ(organ)}"
    entry = _get_cache("vista3d").get(key)
    if entry is None:
        return f"Error: no vista3d segmentation available for organ '{organ}'"
    _queue_images(entry, _VISTA3D_CACHE_DIR,
                  {"organ": organ, "type": "overlay"})
    return entry["text"]


@tool
def vista3d_compare_stations(organ: str) -> str:
    """Compare an organ's volume between dixon_8_W (thorax) and dixon_12_W (abdomen) stations."""
    key = f"{_pid()}::vista3d_compare_stations::{_resolve_organ(organ)}"
    entry = _get_cache("vista3d").get(key)
    if entry is None:
        return f"Error: no vista3d station comparison available for organ '{organ}'"
    _queue_images(entry, _VISTA3D_CACHE_DIR,
                  {"organ": organ, "type": "overlay"})
    return entry["text"]


@tool
def vista3d_compare_visits(organ: str) -> str:
    """Compare an organ's volume between V1 (baseline) and V2 (follow-up) visits. Primary tool for volume change."""
    key = f"{_pid()}::vista3d_compare_visits::{_resolve_organ(organ)}"
    entry = _get_cache("vista3d").get(key)
    if entry is None:
        return f"Error: no vista3d visit comparison available for organ '{organ}'"
    _queue_images(entry, _VISTA3D_CACHE_DIR,
                  {"organ": organ, "type": "overlay"})
    return entry["text"]


_MEDGEMMA_ORGANS = ("brain", "heart", "liver", "pancreas", "wholebody")
_RT_SUFFIX_LEN = len("_report.json")


@tool
def list_available_reports() -> str:
    """List which MedGemma reports (findings/impression/followup) exist for this patient, per organ."""
    man = _load_json(MEDGEMMA_MANIFEST)
    lines = []
    for organ in _MEDGEMMA_ORGANS:
        rp = man.get(f"{_pid()}::{organ}::reports", [])
        if rp:
            rts = sorted(p.split("__")[-1][:-_RT_SUFFIX_LEN] for p in rp)
            lines.append(f"- {organ} ({', '.join(rts)})")
    if not lines:
        return "No MedGemma reports available for this patient."
    return "[Available MedGemma reports]\n" + "\n".join(lines)


@tool
def get_report(organ: str, report_type: str) -> str:
    """Read a MedGemma report. organ: e.g. liver/heart/brain/pancreas/wholebody. report_type: findings|impression|followup."""
    organ = organ.strip().lower()
    rt = report_type.strip().lower()
    man = _load_json(MEDGEMMA_MANIFEST)
    rp = man.get(f"{_pid()}::{organ}::reports", [])
    path = next((p for p in rp if p.endswith(f"__{rt}_report.json")), None)
    if path is None:
        return f"Error: no {rt} report available for organ '{organ}'."
    with open(os.path.join(MEDGEMMA_IMAGE_BASE, path)) as f:
        doc = json.load(f)
    from distill.tool.consultant.build_report_cache import FORMATTERS
    _queue_medgemma_images(organ)
    return FORMATTERS[rt](doc)


@tool
def recall_skills(organ: str = "", keywords: str = "") -> str:
    """Recall reusable skills from the skill library for this patient type. keywords: comma-separated."""
    from distill.tool.consultant.skill_memory_mock_tool import format_skills, match_skills
    store = _load_skill_store(_CACHE_PATHS["skill_store"])
    kw = [k.strip() for k in keywords.split(",") if k.strip()]
    return format_skills(match_skills(store, organ, kw))


@tool
def write_skill(patient_type: str, guidance: str, organ: str = "general",
                pitfalls: str = "", keywords: str = "") -> str:
    """Record a reusable skill after solving the case. patient_type: short description of the patient type; guidance: what worked; pitfalls: what to avoid; keywords: comma-separated."""
    kw = [k.strip() for k in keywords.split(",") if k.strip()]
    skill = {
        "patient_type": patient_type.strip()[:120],
        "organ": organ.strip().lower() or "general",
        "guidance": guidance.strip()[:600],
        "pitfalls": pitfalls.strip()[:400],
        "keywords": kw[:8],
    }
    _thread_local.session_skills.append(skill)
    return (f"Skill recorded for patient type '{skill['patient_type']}'. "
            "It will be available via recall_skills for similar future patients.")


ALL_TOOLS = [list_available_images, view_image, compare_visits,
             vista3d_segment, vista3d_compare_stations, vista3d_compare_visits,
             list_available_reports, get_report, recall_skills, write_skill]
_TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}

_API_SEMAPHORE = None


def set_max_api_concurrent(n: int):
    global _API_SEMAPHORE
    _API_SEMAPHORE = threading.Semaphore(n)


# ── LangGraph nodes ────────────────────────────────────────────────────────

_LLM_MAX_RETRIES = 5


def _build_llm_node(llm_with_tools):
    def llm_node(state):
        last_err = None
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                if _API_SEMAPHORE:
                    with _API_SEMAPHORE:
                        response = llm_with_tools.invoke(state["messages"])
                else:
                    response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}
            except Exception as e:
                last_err = e
                if attempt < _LLM_MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"[MedBoard] LLM call failed ({e.__class__.__name__}: {str(e)[:120]}), "
                          f"retrying in {wait}s (attempt {attempt + 1}/{_LLM_MAX_RETRIES})", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {_LLM_MAX_RETRIES} attempts: {last_err}")
    return llm_node


def _custom_tools_node(state):
    last_message = state["messages"][-1]
    # Hard cap: count tool calls already executed in the conversation.
    used = sum(len(getattr(m, "tool_calls", []) or []) for m in state["messages"]
               if getattr(m, "type", "") == "ai") - len(last_message.tool_calls)
    cap = getattr(_thread_local, "call_cap", MAX_TOOL_CALLS)
    budget_left = cap - used

    tool_messages = []
    for i, tc in enumerate(last_message.tool_calls):
        name, args, tc_id = tc["name"], tc["args"], tc["id"]
        if i >= budget_left:
            tool_messages.append(ToolMessage(
                content=(f"Tool budget exhausted ({cap} calls max). "
                         "Write your final structured report now, including the Answer: line."),
                tool_call_id=tc_id, name=name))
            continue
        t = _TOOL_BY_NAME.get(name)
        try:
            result = t.invoke(args) if t else f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool error: {str(e)[:200]}"
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id, name=name))

    image_queue = _thread_local.image_queue
    if image_queue:
        blocks = [{"type": "image_url",
                   "image_url": {"url": f"data:image/png;base64,{img['b64']}"}}
                  for img in image_queue]
        image_queue.clear()
        tool_messages.append(HumanMessage(content=blocks))
    return {"messages": tool_messages}


def _should_continue(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def _extract_answer_node(state):
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            m = re.search(r"Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)", content, re.IGNORECASE)
            if m:
                letters = sorted(set(l.strip().upper() for l in m.group(1).split(",") if l.strip()))
                return {"answer": ",".join(letters) if letters else "A"}
            break
    return {"answer": "A"}


def build_medboard_agent(model, api_key, base_url, max_tokens=3000):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        base_url=base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=0.0,
        max_tokens=int(os.environ.get("MAX_RESPONSE_TOKENS", max_tokens)),
        extra_body={"enable_thinking": False},
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    graph = StateGraph(MedBoardAgentState)
    graph.add_node("llm", _build_llm_node(llm_with_tools))
    graph.add_node("tools", _custom_tools_node)
    graph.add_node("extract_answer", _extract_answer_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", _should_continue, {"tools": "tools", END: "extract_answer"})
    graph.add_edge("tools", "llm")
    graph.add_edge("extract_answer", END)
    return graph.compile()


def _extract_answer_from_messages(messages):
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            m = re.search(r"Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)", content, re.IGNORECASE)
            if m:
                letters = sorted(set(l.strip().upper() for l in m.group(1).split(",") if l.strip()))
                return ",".join(letters) if letters else "A"
            return "A"
    return "A"


def run_medboard_agent(question, options, patient_id, patient_info,
                       ground_truth=None,
                       model="qwen3.7-plus", api_key=None, base_url=None,
                       max_tokens=3000, recursion_limit=60):
    """Run the medboard agent for one question (two-stage when ground_truth given).

    Stage 1: Phases 1-5, answer extracted from the final message.
    Stage 2 (recording only, needs ground_truth): verdict is injected; if wrong
    the GT is revealed and a Failure Reflection (Phase 6) with write_skill is
    elicited; if correct a short Success Summary skill is elicited.
    """
    _thread_local.patient_id = patient_id
    _thread_local.image_queue = []
    _thread_local.all_images = []
    _thread_local.session_skills = []
    _thread_local.call_cap = MAX_TOOL_CALLS

    agent = build_medboard_agent(model, api_key, base_url, max_tokens)

    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    user_content = (
        f"Patient ID: {patient_id}\n\n"
        f"Patient Info: {str(patient_info)[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Review all evidence sources, verify report claims, then answer in the required 5-phase format."
    )

    from langchain_core.messages import SystemMessage
    initial = [
        SystemMessage(content=SYSTEM_PROMPT.format(max_calls=MAX_TOOL_CALLS)),
        HumanMessage(content=user_content),
    ]
    state1 = agent.invoke(
        {"messages": initial, "patient_id": patient_id,
         "question": question, "options": options, "answer": None},
        config={"recursion_limit": recursion_limit},
    )

    msgs = state1["messages"]
    answer = _extract_answer_from_messages(msgs)
    first_pass_content = ""
    for m in reversed(msgs):
        if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None):
            first_pass_content = m.content if isinstance(m.content, str) else str(m.content)
            break

    reflection_content = None
    correct = None
    if ground_truth is not None:
        gt = str(ground_truth).strip().upper()
        correct = answer == gt
        used_calls = sum(len(getattr(m, "tool_calls", []) or [])
                         for m in msgs if getattr(m, "type", "") == "ai")
        _thread_local.call_cap = used_calls + 2  # write_skill (+1 verification)
        if correct:
            verdict = REFLECTION_PROMPT_CORRECT.format(answer=answer)
        else:
            verdict = REFLECTION_PROMPT_WRONG.format(answer=answer, gt=gt)
        state2 = agent.invoke(
            {"messages": msgs + [HumanMessage(content=verdict)],
             "patient_id": patient_id, "question": question,
             "options": options, "answer": None},
            config={"recursion_limit": recursion_limit},
        )
        msgs = state2["messages"]
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None):
                reflection_content = m.content if isinstance(m.content, str) else str(m.content)
                break

    return {
        "messages": msgs,
        "answer": answer,
        "correct": correct,
        "first_pass_content": first_pass_content,
        "reflection_content": reflection_content,
        "tool_images": _thread_local.all_images,
        "session_skills": _thread_local.session_skills,
    }
