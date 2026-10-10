#!/usr/bin/env python
"""
MedPanel v2 batch trajectory recording — LangGraph agent with tool calling.

Records trajectories where the model autonomously selects imaging modalities,
views slices, and compares visits to answer clinical questions.

Usage:
  python -m distill.tool.radiologist.run_batch \
    --parquet /tmp/mys/dataset/opd_rl_all_0.9/train_dir/train.parquet \
    --mri-dir /mnt/faster_nas/benyou.mys/dataset/ukb_mri_all \
    --vista3d-dir /mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d \
    --vista3d-dir-v2 /mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2 \
    --output /tmp/mys/medpanel_v2_traj \
    --model qwen3.7-plus \
    --max-api-concurrent 500 --max-patient-workers 100
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

from distill.tool.radiologist.radiologist_tool import (
    DEFAULT_MRI_DIR,
    DEFAULT_VISTA3D_DIR,
    DEFAULT_VISTA3D_DIR_V2,
    MAX_TOOL_CALLS,
    run_medpanel_v2_agent,
    set_max_api_concurrent,
)
from distill.tool.common.test_medpanel import parse_question


# ── Parquet data loading ───────────────────────────────────────────────────

def preload_patient_data_batch(parquet_path, target_pids=None):
    """Single-pass read of lightweight columns for multiple patients."""
    pf = pq.ParquetFile(parquet_path)
    available_cols = pf.schema_arrow.names
    cols = ["patient_id", "problems", "solutions"]
    if "narrative_report" in available_cols:
        cols.append("narrative_report")
    has_narrative = "narrative_report" in cols

    pid_to_data = {}
    for batch in pf.iter_batches(batch_size=16, columns=cols):
        table = batch.to_pydict()
        for i, pid in enumerate(table["patient_id"]):
            pid = str(pid)
            if target_pids and pid not in target_pids:
                continue
            problems = table["problems"][i] or []
            solutions = table["solutions"][i] or []
            if not problems:
                continue
            pid_to_data[pid] = {
                "patient_info": table["narrative_report"][i] if has_narrative else "",
                "problems": problems,
                "solutions": solutions,
            }
    return pid_to_data


# ── Message serialization ──────────────────────────────────────────────────

def _save_b64_image(data_uri, image_dir, patient_id, metadata=None):
    """Decode base64 data URI, save to disk. Returns relative path."""
    if data_uri.startswith("data:image/png;base64,"):
        b64_data = data_uri.split(",", 1)[1]
    else:
        b64_data = data_uri

    if metadata:
        modality = metadata.get("modality", "unknown")
        visit = metadata.get("visit", "unknown")
        slice_idx = metadata.get("slice", -1)
        img_type = metadata.get("type", "single")
        if img_type == "comparison" or visit == "comparison":
            fname = f"{modality}_compare_slice{slice_idx}.png"
        else:
            fname = f"{modality}_{visit}_slice{slice_idx}.png"
    else:
        img_hash = hashlib.md5(b64_data.encode()).hexdigest()[:16]
        fname = f"img_{img_hash}.png"

    rel_path = os.path.join("images", patient_id, fname)
    abs_path = os.path.join(image_dir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if not os.path.exists(abs_path):
        with open(abs_path, "wb") as f:
            f.write(base64.b64decode(b64_data))
    return rel_path


def serialize_agent_messages(messages, image_dir, patient_id, tool_images=None):
    """Convert LangGraph messages to OpenAI chat format with tool_calls."""
    b64_to_meta = {}
    if tool_images:
        for img in tool_images:
            b64_to_meta[img["b64"]] = img

    result = []
    for msg in messages:
        if hasattr(msg, "type"):
            msg_type = msg.type
        else:
            msg_type = type(msg).__name__.lower()

        if msg_type == "system":
            result.append({"role": "system", "content": msg.content})
        elif msg_type == "human":
            content = msg.content
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            blocks.append({"type": "text", "text": block["text"]})
                        elif block.get("type") == "image_url":
                            url = block["image_url"]["url"]
                            b64_raw = url.split(",", 1)[1] if "," in url else url
                            meta = b64_to_meta.get(b64_raw)
                            rel = _save_b64_image(url, image_dir, patient_id, metadata=meta)
                            blocks.append({"type": "image", "image": rel})
                    else:
                        blocks.append({"type": "text", "text": str(block)})
                result.append({"role": "user", "content": blocks})
            else:
                result.append({"role": "user", "content": str(content)})
        elif msg_type == "ai":
            entry = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif msg_type == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
                "content": str(msg.content),
            })
    return result


# ── Trajectory conversion ──────────────────────────────────────────────────

PHASE_PATTERNS = {
    "modality_selection": r"## Phase 1:.*?(?=## Phase 2:|$)",
    "v1_analysis": r"## Phase 2:.*?(?=## Phase 3:|$)",
    "v2_analysis": r"## Phase 3:.*?(?=## Phase 4:|$)",
    "insight_synthesis": r"## Phase 4:.*?(?=## Phase 5:|$)",
    "final_diagnosis": r"## Phase 5:.*?$",
}


def _parse_final_response(content):
    """Parse the LLM's 5-phase response into structured data."""
    if not isinstance(content, str):
        content = str(content)
    phases = {}
    for name, pattern in PHASE_PATTERNS.items():
        m = re.search(pattern, content, re.DOTALL)
        phases[name] = m.group(0).strip() if m else ""

    answer_m = re.search(r"Answer:\s*([A-Ia-i](?:\s*,\s*[A-Ia-i])*)", phases.get("final_diagnosis", ""), re.IGNORECASE)
    if answer_m:
        letters = sorted(set(l.strip().upper() for l in answer_m.group(1).split(",") if l.strip()))
        answer = ",".join(letters) if letters else "A"
    else:
        answer = "A"

    insight_m = re.search(r"Clinical Insight:\s*(.+?)(?:\n\n|\n##|$)", phases.get("insight_synthesis", ""), re.DOTALL)
    consensus_insight = insight_m.group(1).strip() if insight_m else ""

    return {"phases": phases, "answer": answer, "consensus_insight": consensus_insight}


def messages_to_dense_trajectory(messages, tool_images, patient_id, question, options, gt, answer, correct, patient_info):
    """Convert agent messages to dense trajectory dict."""
    parsed = _parse_final_response(
        messages[-1].content if messages else ""
    )

    # Save tool images to disk
    image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "outputs", "medpanel_v2_images")
    saved_images = []
    for img in tool_images:
        rel = _save_b64_image(
            f"data:image/png;base64,{img['b64']}",
            image_dir,
            patient_id,
            metadata=img,
        )
        saved_images.append({
            "path": rel,
            "modality": img.get("modality", ""),
            "visit": img.get("visit", ""),
            "slice": img.get("slice", -1),
            "type": img.get("type", "single"),
        })

    return {
        "patient_id": patient_id,
        "question": question,
        "options": options,
        "ground_truth": gt,
        "answer": answer,
        "correct": correct,
        "phases": parsed["phases"],
        "consensus_insight": parsed["consensus_insight"],
        "tool_images": saved_images,
    }


def trajectory_to_sft_messages(trajectory, image_base_dir):
    """Convert dense trajectory to SFT messages (system + user with images + assistant)."""
    from distill.tool.radiologist.radiologist_tool import SYSTEM_PROMPT

    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(max_calls=MAX_TOOL_CALLS)}

    user_blocks = [
        {"type": "text", "text": f"Patient ID: {trajectory['patient_id']}\n\n"
                                  f"Patient Info: {trajectory.get('patient_info', '')[:2000]}\n\n"
                                  f"Question: {trajectory['question']}\n\n"
                                  f"Options:\n" + "\n".join(f"  {k}. {v}" for k, v in trajectory['options'].items())}
    ]

    # Add tool images
    for img in trajectory.get("tool_images", []):
        img_path = os.path.join(image_base_dir, img["path"])
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            label = f"{img.get('modality', '')} {img.get('visit', '')} slice {img.get('slice', '')}"
            user_blocks.append({"type": "text", "text": f"=== {label} ==="})
            user_blocks.append({"type": "image", "image": f"data:image/png;base64,{b64}"})

    user_msg = {"role": "user", "content": user_blocks}

    # Concatenate all phases into single assistant message
    assistant_content = "\n\n".join(trajectory["phases"].values())
    assistant_msg = {"role": "assistant", "content": assistant_content}

    return {
        "messages": [system_msg, user_msg, assistant_msg],
        "metadata": {
            "patient_id": trajectory["patient_id"],
            "answer": trajectory["answer"],
            "ground_truth": trajectory["ground_truth"],
            "correct": trajectory["correct"],
            "consensus_insight": trajectory["consensus_insight"],
        },
    }


# ── Patient processing ─────────────────────────────────────────────────────

def _run_single_question(qidx, question_text, options, gt, patient_id, patient_info,
                         model, api_key, base_url, mri_dir, vista3d_dir, vista3d_dir_v2):
    """Run agent for one question. Thread-safe."""
    try:
        result = run_medpanel_v2_agent(
            question=question_text,
            options=options,
            patient_id=patient_id,
            patient_info=patient_info,
            ground_truth=gt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            mri_dir=mri_dir,
            vista3d_dir=vista3d_dir,
            vista3d_dir_v2=vista3d_dir_v2,
        )
        return qidx, result
    except Exception as e:
        return qidx, {"answer": "A", "correct": gt == "A", "messages": [], "tool_images": [], "error": str(e)}


def process_patient(patient_id, data, output_dir, model, api_key, base_url,
                    mri_dir, vista3d_dir, vista3d_dir_v2,
                    max_workers=5, image_base_dir=None):
    """Process all questions for one patient."""
    problems = data["problems"]
    solutions = data["solutions"]
    patient_info = data["patient_info"]

    questions_data = []
    correct_count = 0
    qidx_to_meta = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for qidx, problem in enumerate(problems):
            parsed = parse_question(problem)
            if isinstance(parsed, tuple):
                question_text, options = parsed
            elif isinstance(parsed, dict):
                question_text = parsed.get("question", "")
                options = parsed.get("options", {})
            else:
                continue
            if not question_text:
                continue
            gt = solutions[qidx] if qidx < len(solutions) else "A"
            qidx_to_meta[qidx] = {"question": question_text, "options": options, "gt": gt}

            futures[pool.submit(
                _run_single_question, qidx, question_text, options, gt,
                patient_id, patient_info, model, api_key, base_url,
                mri_dir, vista3d_dir, vista3d_dir_v2
            )] = qidx

        for fut in as_completed(futures):
            qidx, result = fut.result()
            meta = qidx_to_meta.get(qidx, {})
            answer = result.get("answer", "A")
            correct = result.get("correct", False)
            if correct:
                correct_count += 1

            # Build trajectory
            trajectory = messages_to_dense_trajectory(
                result.get("messages", []),
                result.get("tool_images", []),
                patient_id,
                meta.get("question", ""),
                meta.get("options", {}),
                meta.get("gt", ""),
                answer,
                correct,
                patient_info,
            )
            trajectory["patient_info"] = patient_info
            questions_data.append((qidx, trajectory, result))

    # Sort by question index
    questions_data.sort(key=lambda x: x[0])

    # Write outputs
    dense_path = os.path.join(output_dir, f"dense_{patient_id}.json")
    agent_path = os.path.join(output_dir, f"agent_{patient_id}.jsonl")
    sparse_path = os.path.join(output_dir, f"sparse_{patient_id}.jsonl")

    dense = {
        "patient_id": patient_id,
        "patient_info": patient_info[:2000],
        "questions": [],
    }

    if image_base_dir is None:
        image_base_dir = os.path.join(output_dir, "images")

    with open(agent_path, "w") as af, open(sparse_path, "w") as sf:
        for qidx, traj, result in questions_data:
            # Dense
            dense["questions"].append({
                "index": qidx,
                "question": traj["question"],
                "options": traj["options"],
                "ground_truth": traj["ground_truth"],
                "answer": traj["answer"],
                "correct": traj["correct"],
                "trajectory": traj,
            })

            # Agent (raw tool-calling)
            agent_msg = serialize_agent_messages(
                result.get("messages", []),
                os.path.join(output_dir, "images"),
                patient_id,
                tool_images=result.get("tool_images", []),
            )
            af.write(json.dumps({
                "patient_id": patient_id,
                "question_idx": qidx,
                "question": traj["question"],
                "options": traj["options"],
                "ground_truth": traj["ground_truth"],
                "answer": traj["answer"],
                "correct": traj["correct"],
                "messages": agent_msg,
            }, ensure_ascii=False) + "\n")

            # Sparse (SFT)
            sft = trajectory_to_sft_messages(traj, image_base_dir)
            sf.write(json.dumps(sft, ensure_ascii=False) + "\n")

    with open(dense_path, "w") as f:
        json.dump(dense, f, ensure_ascii=False, indent=2)

    return len(questions_data), correct_count


# ── Batch runner ───────────────────────────────────────────────────────────

def run_batch(parquet_path, output_dir, model, api_key, base_url,
              mri_dir, vista3d_dir, vista3d_dir_v2,
              max_api_concurrent=500, max_patient_workers=100, max_workers=5,
              resume=True, max_patients=0, shard_id=0, num_shards=1):
    """Top-level batch runner."""
    os.makedirs(output_dir, exist_ok=True)
    set_max_api_concurrent(max_api_concurrent)

    # Discover patient IDs from MRI dir
    all_pids = set()
    if os.path.isdir(mri_dir):
        for d in os.listdir(mri_dir):
            if d.endswith("_visit2") or d.endswith("_visit3"):
                all_pids.add(d.split("_visit")[0])
    if os.path.isdir(vista3d_dir):
        for d in os.listdir(vista3d_dir):
            if d.isdigit():
                all_pids.add(d)

    all_pids = sorted(all_pids)
    print(f"Discovered {len(all_pids)} unique patients from filesystem")

    # Sharding
    all_pids = all_pids[shard_id::num_shards]
    if max_patients > 0:
        all_pids = all_pids[:max_patients]

    # Resume filter
    if resume:
        before = len(all_pids)
        all_pids = [p for p in all_pids if not os.path.exists(os.path.join(output_dir, f"dense_{p}.json"))]
        skipped = before - len(all_pids)
        print(f"Skipped {skipped} completed patients (resume)")
    else:
        print("Resume disabled")

    if not all_pids:
        print("No patients to process!")
        return

    # Preload patient data
    print(f"Pre-loading data for {len(all_pids)} patients...")
    t0 = time.time()
    pid_set = set(all_pids)
    patient_data = preload_patient_data_batch(parquet_path, pid_set)
    valid_pids = [p for p in all_pids if p in patient_data]
    print(f"Pre-loaded: {len(valid_pids)} valid in {time.time()-t0:.1f}s "
          f"({len(all_pids) - len(valid_pids)} not in parquet)")

    if not valid_pids:
        print("No valid patients with parquet data!")
        return

    # Process
    print(f"\nProcessing {len(valid_pids)} patients ({max_patient_workers} parallel)...")
    t0 = time.time()
    done = 0
    total_q = 0
    total_correct = 0

    with ThreadPoolExecutor(max_workers=max_patient_workers) as pool:
        futures = {}
        for pid in valid_pids:
            futures[pool.submit(
                process_patient, pid, patient_data[pid], output_dir,
                model, api_key, base_url,
                mri_dir, vista3d_dir, vista3d_dir_v2,
                max_workers
            )] = pid

        for fut in as_completed(futures):
            pid = futures[fut]
            done += 1
            try:
                n_q, n_correct = fut.result()
                total_q += n_q
                total_correct += n_correct
            except Exception as e:
                print(f"  ERROR {pid}: {e}", flush=True)
                n_q, n_correct = 0, 0

            if done % 10 == 0 or done == len(valid_pids):
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                remaining = (len(valid_pids) - done) / (done / elapsed) if done > 0 else 0
                acc = total_correct / total_q * 100 if total_q > 0 else 0
                print(f"  [{done}/{len(valid_pids)}] {rate:.0f} pat/min, "
                      f"ETA {remaining/60:.1f} min, Q={total_q}, Acc={acc:.1f}%",
                      flush=True)

    elapsed = time.time() - t0
    acc = total_correct / total_q * 100 if total_q > 0 else 0
    print(f"\nDone: {done} patients, {total_q} questions, {acc:.1f}% accuracy, "
          f"{elapsed:.0f}s ({elapsed/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="MedPanel v2 batch trajectory recording")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--mri-dir", default=DEFAULT_MRI_DIR)
    parser.add_argument("--vista3d-dir", default=DEFAULT_VISTA3D_DIR)
    parser.add_argument("--vista3d-dir-v2", default=DEFAULT_VISTA3D_DIR_V2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-api-concurrent", type=int, default=500)
    parser.add_argument("--max-patient-workers", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    run_batch(
        parquet_path=args.parquet,
        output_dir=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        mri_dir=args.mri_dir,
        vista3d_dir=args.vista3d_dir,
        vista3d_dir_v2=args.vista3d_dir_v2,
        max_api_concurrent=args.max_api_concurrent,
        max_patient_workers=args.max_patient_workers,
        max_workers=args.max_workers,
        resume=args.resume,
        max_patients=args.max_patients,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
