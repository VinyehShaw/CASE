"""Batch trajectory recording for Vista3D V1/V2 visit comparison agent.

Produces dense JSON (one file per patient) + sparse JSONL (SFT messages)
compatible with existing SFT and RL pipelines.

Usage:
    python -m distill.tool.anatomist.run_batch \
        --parquet /tmp/mys/dataset/opd_rl_all_0.9/train_dir/train.parquet \
        --vista3d-dir /mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d \
        --vista3d-dir-v2 /mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2 \
        --output /mnt/faster_nas/benyou.mys/medixrl/data/vista3d_trajectory \
        --max-patients 10 --model qwen3.6-plus
"""

import os
import sys
import json
import time
import base64
import re
import argparse
from datetime import datetime, timezone

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from distill.tool.anatomist.anatomist_tool import (
    run_vista3d_agent,
    DEFAULT_VISTA3D_DIR,
    DEFAULT_VISTA3D_DIR_V2,
    SYSTEM_PROMPT,
)
from distill.tool.common.test_medpanel import load_patient_data, parse_question


def preload_patient_data_batch(parquet_path: str, target_pids: set) -> dict:
    """Pre-load patient data in a single parquet pass — lightweight (no binary images).

    Only reads patient_id, narrative_report, problems, solutions columns.
    Returns {pid: {"patient_info": str, "problems": list, "solutions": list}}.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    cache = {}
    target_strs = {str(p) for p in target_pids}

    for batch in pf.iter_batches(batch_size=16, columns=["patient_id", "narrative_report", "problems", "solutions"]):
        for i in range(batch.num_rows):
            pid = str(batch["patient_id"][i].as_py())
            if pid not in target_strs:
                continue
            cache[pid] = {
                "patient_info": batch["narrative_report"][i].as_py() or "",
                "problems": batch["problems"][i].as_py() or [],
                "solutions": batch["solutions"][i].as_py() or [],
            }

    return cache


# ============================================================
# LangGraph → Dense JSON conversion
# ============================================================

def _parse_final_response(content: str) -> dict:
    """Parse the 5-phase AI final response into phase dicts.

    Expected format:
        ## Phase 1: Target Selection
        Target: <organ>
        Reasoning: ...

        ## Phase 2: Vista3D V1 Analysis
        ...

        ## Phase 3: Vista3D V2 Analysis
        ...

        ## Phase 4: Insight Synthesis
        ...

        ## Phase 5: Final Diagnosis
        Answer: <letter>
        Reasoning: ...
    """
    phases = {}

    phase_patterns = [
        ("target_selection", r"##\s*Phase\s*1[:\s]*Target Selection\s*\n(.*?)(?=##\s*Phase\s*2|$)", re.DOTALL),
        ("vista3d_v1", r"##\s*Phase\s*2[:\s]*Vista3D V1 Analysis\s*\n(.*?)(?=##\s*Phase\s*3|$)", re.DOTALL),
        ("vista3d_v2", r"##\s*Phase\s*3[:\s]*Vista3D V2 Analysis\s*\n(.*?)(?=##\s*Phase\s*4|$)", re.DOTALL),
        ("insight_synthesis", r"##\s*Phase\s*4[:\s]*Insight Synthesis\s*\n(.*?)(?=##\s*Phase\s*5|$)", re.DOTALL),
        ("final_diagnosis", r"##\s*Phase\s*5[:\s]*Final Diagnosis\s*\n(.*?)$", re.DOTALL),
    ]

    for name, pattern, flags in phase_patterns:
        m = re.search(pattern, content, flags)
        if m:
            text = m.group(1).strip()
            phase_data = {"phase_name": name, "raw_text": text}

            if name == "target_selection":
                t = re.search(r"Target:\s*(.+)", text)
                r = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL)
                if t:
                    phase_data["target"] = t.group(1).strip()
                if r:
                    phase_data["reasoning"] = r.group(1).strip()

            elif name == "final_diagnosis":
                a = re.search(r"Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)", text, re.IGNORECASE)
                r = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL)
                if a:
                    letters = sorted(set(l.strip().upper() for l in a.group(1).split(",") if l.strip()))
                    phase_data["answer"] = ",".join(letters) if letters else "A"
                if r:
                    phase_data["reasoning"] = r.group(1).strip()

            elif name == "insight_synthesis":
                phase_data["consensus_insight"] = text

            phases[name] = phase_data

    return phases


def _extract_tool_data(messages: list) -> dict:
    """Extract tool call data from LangGraph messages.

    Looks for tool_calls in AI messages and corresponding ToolMessage responses.
    Returns dict with organ, v1/v2 volumes, per-station data.
    """
    result = {
        "organ": None,
        "v1_volume_ml": None,
        "v2_volume_ml": None,
        "volume_change_ml": None,
        "volume_change_pct": None,
        "v1_per_station": {},
        "v2_per_station": {},
        "tool_call_type": None,
    }

    for msg in messages:
        # Check for tool calls in AI messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                if tool_name == "vista3d_compare_visits":
                    result["organ"] = tool_args.get("organ", "")
                    result["tool_call_type"] = "compare_visits"
                elif tool_name == "vista3d_segment":
                    result["organ"] = tool_args.get("organ", "")
                    result["tool_call_type"] = "segment"
                elif tool_name == "vista3d_compare_stations":
                    result["organ"] = tool_args.get("organ", "")
                    result["tool_call_type"] = "compare_stations"

        # Parse tool response content
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str) and "V1 vs V2" in content:
                # Parse compare_visits response
                v1_match = re.search(r"V1 \(baseline\):\s*([\d.]+)\s*ml", content)
                v2_match = re.search(r"V2 \(follow-up\):\s*([\d.]+)\s*ml", content)
                change_match = re.search(r"Volume change:\s*([+-]?[\d.]+)\s*ml\s*\(([+-]?[\d.]+)%\)", content)
                if v1_match:
                    result["v1_volume_ml"] = float(v1_match.group(1))
                if v2_match:
                    result["v2_volume_ml"] = float(v2_match.group(1))
                if change_match:
                    result["volume_change_ml"] = float(change_match.group(1))
                    result["volume_change_pct"] = float(change_match.group(2))

                # Parse per-station data
                v1_section = re.search(r"V1 per-station:\n(.*?)(?=V2 per-station|$)", content, re.DOTALL)
                v2_section = re.search(r"V2 per-station:\n(.*?)$", content, re.DOTALL)
                if v1_section:
                    for line in v1_section.group(1).strip().split("\n"):
                        st_match = re.search(r"(dixon_\d+_W):\s*([\d.]+)\s*ml", line.strip())
                        if st_match:
                            result["v1_per_station"][st_match.group(1)] = float(st_match.group(2))
                if v2_section:
                    for line in v2_section.group(1).strip().split("\n"):
                        st_match = re.search(r"(dixon_\d+_W):\s*([\d.]+)\s*ml", line.strip())
                        if st_match:
                            result["v2_per_station"][st_match.group(1)] = float(st_match.group(2))

    return result


def messages_to_dense_trajectory(
    messages: list,
    tool_images: dict,
    patient_id: str,
    question: str,
    options: dict,
    ground_truth: str,
    answer: str,
    correct: bool,
    output_image_dir: str,
) -> dict:
    """Convert LangGraph messages to dense JSON trajectory format.

    Saves overlay images from tool_images to disk and stores relative paths.
    """
    # Extract tool data (organ, volumes, per-station)
    tool_data = _extract_tool_data(messages)

    # Parse final AI response into phases
    final_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_content = msg.content if hasattr(msg, "content") else ""
            break

    parsed_phases = _parse_final_response(final_content)

    # Save overlay images to disk
    overlay_paths = []
    if tool_images:
        os.makedirs(output_image_dir, exist_ok=True)
        for organ, overlays in tool_images.items():
            for visit_key, stations in overlays.items():
                if isinstance(stations, dict):
                    for station, b64 in stations.items():
                        fname = f"{visit_key}_{organ}_{station}_overlay.png"
                        fpath = os.path.join(output_image_dir, fname)
                        try:
                            with open(fpath, "wb") as f:
                                f.write(base64.b64decode(b64))
                            overlay_paths.append(fname)
                        except Exception:
                            pass

    # Build phase list in order
    phases = []

    # Phase 1: Target Selection
    p1 = parsed_phases.get("target_selection", {})
    phases.append({
        "phase_name": "target_selection",
        "target": p1.get("target") or tool_data.get("organ", ""),
        "reasoning": p1.get("reasoning", ""),
    })

    # Phase 2: Vista3D V1 Analysis
    p2 = parsed_phases.get("vista3d_v1", {})
    phases.append({
        "phase_name": "vista3d_v1",
        "organ": tool_data.get("organ", ""),
        "v1_volume_ml": tool_data.get("v1_volume_ml"),
        "v1_per_station": tool_data.get("v1_per_station", {}),
        "raw_text": p2.get("raw_text", ""),
        "overlay_paths": [p for p in overlay_paths if p.startswith("v1_")],
    })

    # Phase 3: Vista3D V2 Analysis
    p3 = parsed_phases.get("vista3d_v2", {})
    phases.append({
        "phase_name": "vista3d_v2",
        "organ": tool_data.get("organ", ""),
        "v2_volume_ml": tool_data.get("v2_volume_ml"),
        "volume_change_ml": tool_data.get("volume_change_ml"),
        "volume_change_pct": tool_data.get("volume_change_pct"),
        "v2_per_station": tool_data.get("v2_per_station", {}),
        "raw_text": p3.get("raw_text", ""),
        "overlay_paths": [p for p in overlay_paths if p.startswith("v2_")],
    })

    # Phase 4: Insight Synthesis
    p4 = parsed_phases.get("insight_synthesis", {})
    phases.append({
        "phase_name": "insight_synthesis",
        "consensus_insight": p4.get("consensus_insight", p4.get("raw_text", "")),
    })

    # Phase 5: Final Diagnosis
    p5 = parsed_phases.get("final_diagnosis", {})
    phases.append({
        "phase_name": "final_diagnosis",
        "answer": answer or p5.get("answer", ""),
        "reasoning": p5.get("reasoning", ""),
    })

    final_result = {
        "answer": answer,
        "ground_truth": ground_truth,
        "correct": correct,
        "consensus_insight": p4.get("consensus_insight", p4.get("raw_text", "")),
        "target": tool_data.get("organ", ""),
        "v1_volume_ml": tool_data.get("v1_volume_ml"),
        "v2_volume_ml": tool_data.get("v2_volume_ml"),
        "volume_change_ml": tool_data.get("volume_change_ml"),
        "volume_change_pct": tool_data.get("volume_change_pct"),
    }

    return {
        "phases": phases,
        "final_result": final_result,
    }


def trajectory_to_sft_messages(trajectory: dict, patient_info: str, question: str, options: dict, overlay_image_dir: str) -> list:
    """Convert dense trajectory to SFT messages (prompt + completion).

    In multimodal mode, overlay images are loaded from disk and injected as
    image content blocks in the user message.
    """
    opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    # Build user message with image placeholders
    user_text = (
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_text}\n\n"
    )

    # Collect overlay images from phases
    image_blocks = []
    for phase in trajectory.get("phases", []):
        for fname in phase.get("overlay_paths", []):
            fpath = os.path.join(overlay_image_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                visit_label = "V1" if fname.startswith("v1_") else "V2"
                image_blocks.append({
                    "type": "text",
                    "text": f"=== {visit_label} Overlay ({fname.split('_', 2)[-1].replace('_overlay.png', '')}) ===",
                })
                image_blocks.append({
                    "type": "image",
                    "image": f"data:image/png;base64,{b64}",
                })

    # Build SFT messages
    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT,
    })

    # User message (with images if available)
    if image_blocks:
        user_content = [{"type": "text", "text": user_text}] + image_blocks
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_text})

    # Assistant completion — concatenate all 5 phases
    phase_texts = []
    phase_titles = {
        "target_selection": "## Phase 1: Target Selection",
        "vista3d_v1": "## Phase 2: Vista3D V1 Analysis",
        "vista3d_v2": "## Phase 3: Vista3D V2 Analysis",
        "insight_synthesis": "## Phase 4: Insight Synthesis",
        "final_diagnosis": "## Phase 5: Final Diagnosis",
    }

    for phase in trajectory.get("phases", []):
        title = phase_titles.get(phase["phase_name"], phase["phase_name"])
        raw = phase.get("raw_text", "")

        if phase["phase_name"] == "target_selection":
            raw = f"Target: {phase.get('target', '')}\nReasoning: {phase.get('reasoning', '')}"
        elif phase["phase_name"] == "insight_synthesis":
            raw = phase.get("consensus_insight", "")
        elif phase["phase_name"] == "final_diagnosis":
            raw = f"Answer: {phase.get('answer', '')}\nReasoning: {phase.get('reasoning', '')}"
        elif phase["phase_name"] == "vista3d_v1":
            v1 = phase.get("v1_volume_ml")
            stations = phase.get("v1_per_station", {})
            station_text = "\n".join(f"  {k}: {v:.1f} ml" for k, v in stations.items()) if stations else "  (no station data)"
            raw = f"Organ: {phase.get('organ', '')}\nV1 Volume: {v1:.1f} ml" if v1 else f"Organ: {phase.get('organ', '')}\n(not found)"
            raw += f"\nPer-station:\n{station_text}"
            if raw and phase.get("raw_text"):
                raw += f"\n{phase['raw_text']}"
        elif phase["phase_name"] == "vista3d_v2":
            v2 = phase.get("v2_volume_ml")
            change = phase.get("volume_change_ml")
            change_pct = phase.get("volume_change_pct")
            stations = phase.get("v2_per_station", {})
            station_text = "\n".join(f"  {k}: {v:.1f} ml" for k, v in stations.items()) if stations else "  (no station data)"
            raw = f"Organ: {phase.get('organ', '')}\nV2 Volume: {v2:.1f} ml" if v2 else f"Organ: {phase.get('organ', '')}\n(not found)"
            if change is not None:
                raw += f"\nVolume change: {change:+.1f} ml ({change_pct:+.1f}%)"
            raw += f"\nPer-station:\n{station_text}"
            if phase.get("raw_text"):
                raw += f"\n{phase['raw_text']}"

        phase_texts.append(f"{title}\n{raw}")

    messages.append({
        "role": "assistant",
        "content": "\n\n".join(phase_texts),
    })

    return messages


# ============================================================
# Raw agent message serialization (tool-calling trajectory)
# ============================================================

def _save_b64_image(data_uri: str, image_dir: str, patient_id: str) -> str | None:
    """Decode a base64 data URI, save to image_dir with content-hash name (dedup).

    Returns relative path 'images/{pid}/img_{hash}.png' or None on failure.
    """
    import hashlib
    try:
        header, b64 = data_uri.split(",", 1)
        ext = "png" if "png" in header else "jpg"
        raw = base64.b64decode(b64)
    except Exception:
        return None
    h = hashlib.md5(raw).hexdigest()[:16]
    fname = f"img_{h}.{ext}"
    fpath = os.path.join(image_dir, fname)
    if not os.path.exists(fpath):
        os.makedirs(image_dir, exist_ok=True)
        with open(fpath, "wb") as f:
            f.write(raw)
    return f"images/{patient_id}/{fname}"


def _serialize_content_blocks(content, image_dir: str, patient_id: str):
    """Serialize message content: keep text blocks, save image blocks to disk."""
    if isinstance(content, str):
        return content
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:image/"):
                rel = _save_b64_image(url, image_dir, patient_id)
                if rel:
                    blocks.append({"type": "image", "image": rel})
    return blocks


def serialize_agent_messages(messages: list, image_dir: str, patient_id: str) -> list:
    """Convert LangChain messages to OpenAI chat format with tool_calls.

    Image data URIs are saved to disk and replaced with relative paths.
    Output schema per message:
      system/user: {"role", "content": str | blocks}
      assistant:   {"role", "content": str, "tool_calls"?: [{id, type, function:{name, arguments}}]}
      tool:        {"role", "tool_call_id", "name", "content": str}
    """
    out = []
    for msg in messages:
        mtype = getattr(msg, "type", None)
        if mtype == "system":
            out.append({"role": "system", "content": msg.content if isinstance(msg.content, str) else str(msg.content)})
        elif mtype == "human":
            out.append({"role": "user", "content": _serialize_content_blocks(msg.content, image_dir, patient_id)})
        elif mtype == "ai":
            entry = {"role": "assistant"}
            content = msg.content
            entry["content"] = content if isinstance(content, str) else _serialize_content_blocks(content, image_dir, patient_id)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            out.append(entry)
        elif mtype == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": getattr(msg, "tool_call_id", ""),
                "name": getattr(msg, "name", "") or "",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
    return out


# ============================================================
# Batch Runner
# ============================================================

def _run_single_question(
    qidx: int,
    question: str,
    options: dict,
    gt: str,
    patient_id: str,
    patient_info: str,
    vista3d_dir: str,
    vista3d_dir_v2: str,
    model: str,
    image_dir: str,
) -> dict:
    """Run agent for a single question and convert to trajectory. Thread-safe."""
    try:
        result = run_vista3d_agent(
            question=question,
            options=options,
            patient_id=patient_id,
            patient_info=patient_info,
            ground_truth=gt,
            vista3d_dir=vista3d_dir,
            vista3d_dir_v2=vista3d_dir_v2,
            model=model,
            include_preview_images=False,
        )
    except Exception as e:
        return {
            "qidx": qidx,
            "question": question,
            "options": options,
            "ground_truth": gt,
            "answer": None,
            "correct": None,
            "error": str(e),
            "trajectory": None,
            "sft_line": None,
            "agent_line": None,
        }

    answer = result.get("answer", "")
    correct = result.get("correct")
    messages = result.get("messages", [])
    tool_images = result.get("tool_images", {})

    traj = messages_to_dense_trajectory(
        messages=messages,
        tool_images=tool_images,
        patient_id=patient_id,
        question=question,
        options=options,
        ground_truth=gt,
        answer=answer,
        correct=correct,
        output_image_dir=image_dir,
    )

    sft_msgs = trajectory_to_sft_messages(
        trajectory=traj,
        patient_info=patient_info,
        question=question,
        options=options,
        overlay_image_dir=image_dir,
    )
    sft_entry = {
        "messages": sft_msgs,
        "metadata": {
            "patient_id": patient_id,
            "question_idx": qidx,
            "answer": answer,
            "ground_truth": gt,
            "correct": correct,
            "consensus_insight": traj["final_result"].get("consensus_insight", ""),
            "target": traj["final_result"].get("target", ""),
            "v1_volume_ml": traj["final_result"].get("v1_volume_ml"),
            "v2_volume_ml": traj["final_result"].get("v2_volume_ml"),
        },
    }

    agent_msgs = serialize_agent_messages(messages, image_dir, patient_id)
    agent_entry = {
        "patient_id": patient_id,
        "question_idx": qidx,
        "question": question,
        "options": options,
        "ground_truth": gt,
        "answer": answer,
        "correct": correct,
        "messages": agent_msgs,
    }

    return {
        "qidx": qidx,
        "question": question,
        "options": options,
        "ground_truth": gt,
        "answer": answer,
        "correct": correct,
        "trajectory": traj,
        "sft_line": json.dumps(sft_entry, ensure_ascii=False),
        "agent_line": json.dumps(agent_entry, ensure_ascii=False),
    }


def process_patient(
    patient_id: str,
    data: dict,
    parquet_path: str,
    vista3d_dir: str,
    vista3d_dir_v2: str,
    output_dir: str,
    model: str,
    max_workers: int = 5,
) -> dict:
    """Process all questions for a single patient (concurrent within patient).

    Returns:
        {patient_id, total, correct, errors}
    """
    patient_info = data.get("patient_info", "")
    problems = data.get("problems", [])
    solutions = data.get("solutions", [])

    image_dir = os.path.join(output_dir, "images", patient_id)

    # Parse all questions first
    parsed = []
    for qidx, problem in enumerate(problems):
        question, options = parse_question(problem)
        if not question:
            continue
        gt = solutions[qidx] if qidx < len(solutions) else ""
        parsed.append((qidx, question, options, gt))

    # Run questions concurrently
    results = []
    if max_workers > 1 and len(parsed) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_question, qidx, q, o, gt,
                    patient_id, patient_info, vista3d_dir, vista3d_dir_v2,
                    model, image_dir,
                ): qidx
                for qidx, q, o, gt in parsed
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for qidx, q, o, gt in parsed:
            results.append(_run_single_question(
                qidx, q, o, gt, patient_id, patient_info,
                vista3d_dir, vista3d_dir_v2, model, image_dir,
            ))

    # Sort by question index
    results.sort(key=lambda r: r["qidx"])

    # Build output
    questions = []
    sparse_lines = []
    agent_lines = []
    for r in results:
        q_entry = {
            "index": r["qidx"],
            "question": r["question"],
            "options": r["options"],
            "ground_truth": r["ground_truth"],
            "answer": r["answer"],
            "correct": r["correct"],
            "trajectory": r["trajectory"],
        }
        if r.get("error"):
            q_entry["error"] = r["error"]
        questions.append(q_entry)

        if r.get("sft_line"):
            sparse_lines.append(r["sft_line"])
        if r.get("agent_line"):
            agent_lines.append(r["agent_line"])

        status = "✓" if r["correct"] else "✗"
        err = f" ERR:{r['error'][:40]}" if r.get("error") else ""
        print(f"  Q{r['qidx']}: {r['answer']} (GT:{r['ground_truth']}) {status}{err}")

    # Save dense JSON
    dense_path = os.path.join(output_dir, f"dense_{patient_id}.json")
    dense_data = {
        "patient_id": patient_id,
        "patient_info": patient_info[:2000],
        "modalities": ["Whole-Body Dixon MRI"],
        "questions": questions,
    }
    with open(dense_path, "w", encoding="utf-8") as f:
        json.dump(dense_data, f, ensure_ascii=False, indent=2)

    # Save sparse JSONL
    sparse_path = os.path.join(output_dir, f"sparse_{patient_id}.jsonl")
    with open(sparse_path, "w", encoding="utf-8") as f:
        for line in sparse_lines:
            f.write(line + "\n")

    # Save agent JSONL (raw tool-calling trajectory for tool-use SFT/RL)
    agent_path = os.path.join(output_dir, f"agent_{patient_id}.jsonl")
    with open(agent_path, "w", encoding="utf-8") as f:
        for line in agent_lines:
            f.write(line + "\n")

    total = len(questions)
    correct_count = sum(1 for q in questions if q.get("correct"))
    error_count = sum(1 for q in questions if q.get("error"))

    return {
        "patient_id": patient_id,
        "total": total,
        "correct": correct_count,
        "errors": error_count,
    }


def run_batch(
    parquet_path: str,
    vista3d_dir: str = DEFAULT_VISTA3D_DIR,
    vista3d_dir_v2: str = None,
    output_dir: str = "/tmp/vista3d_trajectory",
    max_patients: int = None,
    model: str = "qwen3.6-plus",
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    max_workers: int = 5,
    max_api_concurrent: int = 500,
    max_patient_workers: int = 100,
):
    """Run Vista3D V1/V2 trajectory recording on multiple patients.

    Args:
        parquet_path: Path to patient data parquet
        vista3d_dir: Path to V1 (baseline) Vista3D data
        vista3d_dir_v2: Path to V2 (follow-up) Vista3D data
        output_dir: Output directory for trajectories
        max_patients: Limit number of patients (for testing)
        model: LLM model name
        shard_id: Shard ID for multi-GPU parallelism
        num_shards: Total number of shards
        resume: Skip patients with existing dense JSON
        max_workers: Concurrent question workers per patient
        max_api_concurrent: Global API concurrency limit (semaphore)
        max_patient_workers: Concurrent patients to process
    """
    from distill.tool.anatomist.anatomist_tool import set_max_api_concurrent

    set_max_api_concurrent(max_api_concurrent)
    print(f"API concurrency limit: {max_api_concurrent}")
    print(f"Patient workers: {max_patient_workers}")
    print(f"Question workers/patient: {max_workers}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(vista3d_dir):
        print(f"Error: V1 Vista3D directory not found: {vista3d_dir}")
        return

    # Get patient IDs from V1 directory
    patient_ids = sorted([d for d in os.listdir(vista3d_dir) if d.isdigit()])

    # Apply sharding
    if num_shards > 1:
        patient_ids = patient_ids[shard_id::num_shards]
        print(f"Shard {shard_id}/{num_shards}: {len(patient_ids)} patients")

    print(f"Vista3D-dir patients (before parquet intersect): {len(patient_ids)}")

    # Resume: filter out completed patients
    skipped = 0
    if resume:
        filtered = []
        for pid in patient_ids:
            dense_path = os.path.join(output_dir, f"dense_{pid}.json")
            if os.path.exists(dense_path):
                skipped += 1
            else:
                filtered.append(pid)
        patient_ids = filtered
        print(f"Skipped {skipped} completed patients (resume)")

    if not patient_ids:
        print("No patients to process.")
        return

    # Pre-load all patient data in a single parquet pass (lightweight — no binary images)
    print(f"Pre-loading data for {len(patient_ids)} patients...")
    t0 = time.time()
    patient_data_cache = preload_patient_data_batch(parquet_path, set(patient_ids))
    patient_ids = [pid for pid in patient_ids if pid in patient_data_cache]
    print(f"Pre-loaded: {len(patient_data_cache)} valid in {time.time()-t0:.1f}s")

    # Cap AFTER the parquet intersect, so max_patients limits EVALUABLE (in-bank)
    # patients. Capping before would pick the smallest vista3d-dir ids, which are
    # usually not in the eval parquet -> 0 valid.
    if max_patients:
        patient_ids = patient_ids[:max_patients]
    print(f"Total patients to process: {len(patient_ids)}")

    if not patient_ids:
        print("No patients with valid data to process.")
        return

    # Progress tracking
    progress = {"done": 0, "total": len(patient_ids), "start": time.time(), "correct": 0, "questions": 0}

    def progress_bar():
        while progress["done"] < progress["total"]:
            elapsed = time.time() - progress["start"]
            done = progress["done"]
            total = progress["total"]
            rate = done / max(elapsed, 1)
            eta = (total - done) / max(rate, 0.01) if done > 0 else 0
            bar_len = 20
            filled = int(bar_len * done / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            acc = f" acc={progress['correct']}/{progress['questions']}" if progress["questions"] > 0 else ""
            sys.stderr.write(
                f"\r[{bar}] {done}/{total} ({rate:.1f} pat/s, ETA {eta:.0f}s){acc}  "
            )
            sys.stderr.flush()
            time.sleep(5)

    import threading
    progress_thread = threading.Thread(target=progress_bar, daemon=True)
    progress_thread.start()

    all_results = []
    results_lock = threading.Lock()

    def _process_one_patient(i, pid):
        """Process a single patient — designed for ThreadPoolExecutor."""
        data = patient_data_cache.get(pid)
        if data is None:
            with results_lock:
                progress["done"] += 1
            return None

        try:
            stats = process_patient(
                patient_id=pid,
                data=data,
                parquet_path=parquet_path,
                vista3d_dir=vista3d_dir,
                vista3d_dir_v2=vista3d_dir_v2,
                output_dir=output_dir,
                model=model,
                max_workers=max_workers,
            )
            with results_lock:
                all_results.append(stats)
                progress["correct"] += stats["correct"]
                progress["questions"] += stats["total"]
                progress["done"] += 1
            return stats
        except Exception as e:
            print(f"  [Error] Patient {pid} failed: {e}")
            with results_lock:
                progress["done"] += 1
            return None

    # Run patients concurrently
    import concurrent.futures
    if max_patient_workers > 1 and len(patient_ids) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_patient_workers) as executor:
            futures = {
                executor.submit(_process_one_patient, i, pid): pid
                for i, pid in enumerate(patient_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Exceptions already handled in _process_one_patient
    else:
        for i, pid in enumerate(patient_ids):
            _process_one_patient(i, pid)

    # Stop progress bar
    progress["done"] = progress["total"]
    time.sleep(1)

    # Summary
    total_patients = len(all_results)
    total_questions = sum(r["total"] for r in all_results)
    total_correct = sum(r["correct"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"Vista3D V1/V2 Trajectory Recording Complete")
    print(f"{'='*60}")
    print(f"  Patients processed: {total_patients}")
    print(f"  Patients skipped (resume): {skipped}")
    print(f"  Questions: {total_questions}")
    print(f"  Correct: {total_correct}")
    if total_questions > 0:
        print(f"  Accuracy: {total_correct/total_questions*100:.1f}%")
    if total_errors > 0:
        print(f"  Errors: {total_errors}")
    print(f"  Output: {output_dir}")
    print(f"  Dense: {output_dir}/dense_*.json")
    print(f"  Sparse: {output_dir}/sparse_*.jsonl")
    print(f"  Images: {output_dir}/images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vista3D V1/V2 Trajectory Recording")
    parser.add_argument("--parquet", type=str,
                        default="/tmp/mys/dataset/opd_rl_all_0.9/train_dir/train.parquet")
    parser.add_argument("--vista3d-dir", type=str, default=DEFAULT_VISTA3D_DIR)
    parser.add_argument("--vista3d-dir-v2", type=str, default=None,
                        help="Path to V2 (follow-up) Vista3D data")
    parser.add_argument("--output", type=str,
                        default="/mnt/faster_nas/benyou.mys/medixrl/data/vista3d_trajectory")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen3.6-plus")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Concurrent question workers per patient")
    parser.add_argument("--max-api-concurrent", type=int, default=500,
                        help="Global API concurrency limit (semaphore)")
    parser.add_argument("--max-patient-workers", type=int, default=100,
                        help="Concurrent patients to process")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume (reprocess all)")
    args = parser.parse_args()

    run_batch(
        parquet_path=args.parquet,
        vista3d_dir=args.vista3d_dir,
        vista3d_dir_v2=args.vista3d_dir_v2,
        output_dir=args.output,
        max_patients=args.max_patients,
        model=args.model,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        resume=not args.no_resume,
        max_workers=args.max_workers,
        max_api_concurrent=args.max_api_concurrent,
        max_patient_workers=args.max_patient_workers,
    )
