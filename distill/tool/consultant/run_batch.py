#!/usr/bin/env python3
"""medboard_run_batch.py — record medboard integrated-review trajectories.

Small-batch / full recording of the medboard agent (cache-backed tools, so
GPU-free and deterministic). Output schema matches medpanel_v2/vista3d
recordings so downstream SFT/RL prep can reuse them:

    {output_dir}/agent_{pid}.jsonl    # raw tool-calling trajectory per question
    {output_dir}/dense_{pid}.json     # parsed phases + metadata
    {output_dir}/skills_{pid}.jsonl   # skills written during recording
    {output_dir}/images/{pid}/        # copied tool images

Usage (demo):
    python -m distill.tool.consultant.run_batch \
        --max-patients 3 --max-questions 2 --model qwen3.7-plus \
        --output-dir /mnt/faster_nas/benyou.mys/dataset/medboard_trajectory_demo
"""
import argparse
import base64
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from distill.tool.common.test_medpanel import parse_question  # noqa: E402
from distill.tool.consultant.consultant_tool import (  # noqa: E402
    SYSTEM_PROMPT, MAX_TOOL_CALLS, init_medboard_caches, run_medboard_agent,
    set_max_api_concurrent, DEFAULT_REPORT_CACHE,
)

DEFAULT_QUESTION_BANK = os.environ.get(
    "CASE_EVAL_BANK", "/mnt/faster_nas/benyou.mys/medixrl/data/eval_bank_unified_0811_fix.parquet")

PHASE_PATTERNS = {
    "skill_recall_plan": r"## Phase 1:.*?(?=## Phase 2:|$)",
    "multi_source_review": r"## Phase 2:.*?(?=## Phase 3:|$)",
    "doubt_verification": r"## Phase 3:.*?(?=## Phase 4:|$)",
    "reflective_insight": r"## Phase 4:.*?(?=## Phase 5:|$)",
    "final_answer": r"## Phase 5:.*?$",
    "reflection": r"## Phase 6:.*?$",
}


# ── Serialization ──────────────────────────────────────────────────────────

_img_counter = {}
_img_counter_lock = threading.Lock()


def _save_tool_image(meta, image_dir, patient_id):
    """Copy a cache image into the output image dir; return relative path."""
    src = meta.get("src_path_abs")
    rel_out = None
    if src and os.path.exists(src):
        with _img_counter_lock:
            key = (patient_id, src)
            n = _img_counter.get(key, 0)
            _img_counter[key] = n + 1
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", meta.get("label", "img"))[:40]
        rel_out = f"{patient_id}/{base}_{n}.png"
        os.makedirs(os.path.dirname(os.path.join(image_dir, rel_out)), exist_ok=True)
        with open(src, "rb") as fin, open(os.path.join(image_dir, rel_out), "wb") as fout:
            fout.write(fin.read())
        return rel_out
    # Fallback: decode base64
    with _img_counter_lock:
        n = _img_counter.get(("b64", patient_id), 0)
        _img_counter[("b64", patient_id)] = n + 1
    rel_out = f"{patient_id}/img_{n}.png"
    os.makedirs(os.path.dirname(os.path.join(image_dir, rel_out)), exist_ok=True)
    with open(os.path.join(image_dir, rel_out), "wb") as f:
        f.write(base64.b64decode(meta["b64"]))
    return rel_out


def serialize_agent_messages(messages, image_dir, patient_id, image_meta_by_b64):
    """Convert LangGraph messages to OpenAI chat format (same schema as
    medpanel_v2 run_batch.serialize_agent_messages)."""
    result = []
    for msg in messages:
        msg_type = getattr(msg, "type", type(msg).__name__.lower())
        if msg_type == "system":
            result.append({"role": "system", "content": msg.content})
        elif msg_type == "human":
            content = msg.content
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        b64 = url.split(",", 1)[1] if "," in url else url
                        meta = image_meta_by_b64.get(b64, {"b64": b64})
                        rel = _save_tool_image(meta, image_dir, patient_id)
                        blocks.append({"type": "image", "image": rel})
                    elif isinstance(block, dict) and block.get("type") == "text":
                        blocks.append({"type": "text", "text": block["text"]})
                    else:
                        blocks.append({"type": "text", "text": str(block)})
                result.append({"role": "user", "content": blocks})
            else:
                result.append({"role": "user", "content": str(content)})
        elif msg_type == "ai":
            entry = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "tool_calls", None):
                entry["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif msg_type == "tool":
            result.append({"role": "tool", "tool_call_id": msg.tool_call_id,
                           "name": msg.name, "content": str(msg.content)})
    return result


def _parse_phases(content):
    if not isinstance(content, str):
        content = str(content)
    phases = {}
    for name, pattern in PHASE_PATTERNS.items():
        m = re.search(pattern, content, re.DOTALL)
        phases[name] = m.group(0).strip() if m else ""
    return phases


# ── Patient processing ─────────────────────────────────────────────────────

def _run_single_question(qidx, question_text, options, gt, patient_id, patient_info,
                         model, api_key, base_url):
    try:
        result = run_medboard_agent(
            question=question_text, options=options, patient_id=patient_id,
            patient_info=patient_info, ground_truth=gt,
            model=model, api_key=api_key, base_url=base_url,
        )
        return qidx, result
    except Exception as e:
        return qidx, {"answer": "A", "messages": [], "tool_images": [],
                      "session_skills": [], "error": str(e)[:300]}


def process_patient(patient_id, data, output_dir, model, api_key, base_url,
                    max_questions=0, max_workers=2):
    problems = data["problems"]
    solutions = data["solutions"]
    patient_info = data.get("narrative_report", "")

    image_dir = os.path.join(output_dir, "images")
    questions_out = []
    skills_out = []
    correct_count = 0

    jobs = []
    for qidx, problem in enumerate(problems):
        if max_questions and len(jobs) >= max_questions:
            break
        parsed = parse_question(problem)
        if isinstance(parsed, tuple):
            question_text, options = parsed
        elif isinstance(parsed, dict):
            question_text, options = parsed.get("question", ""), parsed.get("options", {})
        else:
            continue
        if not question_text or not options:
            continue
        gt = solutions[qidx] if qidx < len(solutions) else "A"
        jobs.append((qidx, question_text, options, str(gt).strip().upper()))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_single_question, qidx, qt, opts, gt,
                               patient_id, patient_info, model, api_key, base_url):
                   (qidx, qt, opts, gt) for qidx, qt, opts, gt in jobs}
        for fut in as_completed(futures):
            qidx, qt, opts, gt = futures[fut]
            _, result = fut.result()
            answer = result.get("answer", "A")
            correct = result.get("correct")
            if correct is None:
                correct = answer == gt
            correct_count += int(correct)

            # Attach absolute src paths to image meta for copying
            image_meta_by_b64 = {}
            for meta in result.get("tool_images", []):
                m = dict(meta)
                if m.get("src_path"):
                    for cache_dir in (
                            os.environ.get("MEDPANEL_V2_TOOL_CACHE_DIR", "/mnt/faster_nas/benyou.mys/medixrl/data/medpanel_v2_tool_cache"),
                            os.environ.get("VISTA3D_TOOL_CACHE_DIR", "/mnt/faster_nas/benyou.mys/medixrl/data/vista3d_tool_cache")):
                        cand = os.path.join(cache_dir, m["src_path"])
                        if os.path.exists(cand):
                            m["src_path_abs"] = cand
                            break
                image_meta_by_b64[m["b64"]] = m

            messages = serialize_agent_messages(
                result.get("messages", []), image_dir, patient_id, image_meta_by_b64)

            phases = _parse_phases(result.get("first_pass_content", ""))
            phases.pop("reflection", None)
            phases["reflection"] = (result.get("reflection_content") or "").strip()

            questions_out.append({
                "question_idx": qidx,
                "question": qt, "options": opts, "ground_truth": gt,
                "answer": answer, "correct": correct,
                "phases": phases,
                "error": result.get("error", ""),
                "_messages": messages,
            })
            skill_source = "success" if correct else "failure_reflection"
            for sk in result.get("session_skills", []):
                skills_out.append({"patient_id": patient_id, "question_idx": qidx,
                                   "source": skill_source, **sk})

    questions_out.sort(key=lambda x: x["question_idx"])

    # Write agent_*.jsonl
    with open(os.path.join(output_dir, f"agent_{patient_id}.jsonl"), "w") as f:
        for q in questions_out:
            f.write(json.dumps({
                "patient_id": patient_id,
                "data_source": "medboard_opsd",
                "question_idx": q["question_idx"],
                "question": q["question"], "options": q["options"],
                "ground_truth": q["ground_truth"],
                "answer": q["answer"], "correct": q["correct"],
                "messages": q["_messages"],
            }, ensure_ascii=False) + "\n")

    # Write dense_*.json
    dense = {"patient_id": patient_id, "patient_info": str(patient_info)[:2000],
             "questions": [{k: v for k, v in q.items() if k != "_messages"} for q in questions_out]}
    with open(os.path.join(output_dir, f"dense_{patient_id}.json"), "w") as f:
        json.dump(dense, f, ensure_ascii=False, indent=2)

    # Write sparse_*.jsonl (extra preserved SFT-format copy, same shape as vista3d)
    with open(os.path.join(output_dir, f"sparse_{patient_id}.jsonl"), "w") as f:
        for q in questions_out:
            f.write(json.dumps({
                "messages": q["_messages"],
                "metadata": {"patient_id": patient_id, "data_source": "medboard_opsd",
                             "question_idx": q["question_idx"], "question": q["question"],
                             "options": q["options"], "ground_truth": q["ground_truth"],
                             "answer": q["answer"], "correct": q["correct"]},
            }, ensure_ascii=False) + "\n")

    # Write skills_*.jsonl
    if skills_out:
        with open(os.path.join(output_dir, f"skills_{patient_id}.jsonl"), "w") as f:
            f.write(json.dumps({"patient_id": patient_id, "skills": skills_out},
                               ensure_ascii=False) + "\n")

    return len(questions_out), correct_count, len(skills_out)


# ── Main ───────────────────────────────────────────────────────────────────

def load_question_bank(parquet_path, wanted_pids=None):
    import time
    import pandas as pd
    cols = ["patient_id", "problems", "solutions", "narrative_report"]
    df = None
    for attempt in range(5):
        try:
            df = pd.read_parquet(parquet_path, columns=cols)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"[load_question_bank] read attempt {attempt + 1} failed ({e}); retrying...")
            time.sleep(2)
    bank = {}
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        if wanted_pids and pid not in wanted_pids:
            continue
        bank[pid] = {"problems": list(row["problems"]),
                     "solutions": list(row["solutions"]),
                     "narrative_report": str(row.get("narrative_report", ""))}
    return bank


def pick_demo_patients(report_cache_path, bank, max_patients):
    """Patients present in BOTH the question bank and the report cache."""
    with open(report_cache_path) as f:
        cache = json.load(f)
    report_pids = {k.split("::")[0] for k in cache if k.endswith("::list_available_reports")}
    both = sorted(set(bank) & report_pids)
    return both[:max_patients]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-bank", default=DEFAULT_QUESTION_BANK)
    ap.add_argument("--patient-ids", default="", help="comma-separated; empty = auto-pick")
    ap.add_argument("--max-patients", type=int, default=3)
    ap.add_argument("--max-questions", type=int, default=2)
    ap.add_argument("--model", default="qwen3.7-plus")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--output-dir",
                    default="/mnt/faster_nas/benyou.mys/dataset/medboard_trajectory_demo")
    ap.add_argument("--report-cache", default=DEFAULT_REPORT_CACHE)
    ap.add_argument("--skill-store", default="",
                    help="override the skill store path (empty = recording-side default)")
    ap.add_argument("--max-api-concurrent", type=int, default=500)
    ap.add_argument("--patient-workers", type=int, default=100,
                    help="patients recorded in parallel (same pattern as medpanel run_batch)")
    ap.add_argument("--question-workers", type=int, default=2,
                    help="questions per patient in parallel")
    ap.add_argument("--auto-resume-rounds", type=int, default=5,
                    help="re-submit patients whose agent_*.jsonl is still missing, up to N rounds")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_max_api_concurrent(args.max_api_concurrent)
    if args.skill_store:
        init_medboard_caches(report_cache=args.report_cache,
                             skill_store=args.skill_store)
    else:
        init_medboard_caches(report_cache=args.report_cache)

    wanted = set(x.strip() for x in args.patient_ids.split(",") if x.strip()) or None
    bank = load_question_bank(args.question_bank, wanted)
    if wanted:
        pids = sorted(wanted & set(bank))[:args.max_patients]
    else:
        # Unified NAS source: patients with ::reports entries in the manifest.
        from distill.tool.consultant.consultant_tool import MEDGEMMA_MANIFEST
        with open(MEDGEMMA_MANIFEST) as f:
            man = json.load(f)
        have = {k.split("::")[0] for k in man if k.endswith("::reports")}
        pids = sorted(set(bank) & have)[:args.max_patients]
    print(f"[medboard] recording {len(pids)} patients x "
          f"{'all' if args.max_questions == 0 else f'<= {args.max_questions}'} questions, "
          f"model={args.model}", flush=True)

    total_q = total_correct = total_skills = 0
    done = 0
    lock = threading.Lock()

    def missing_pids():
        return [pid for pid in pids
                if not os.path.exists(os.path.join(args.output_dir, f"agent_{pid}.jsonl"))]

    missing = missing_pids()
    skipped = len(pids) - len(missing)
    done = skipped
    if skipped:
        print(f"[medboard] {skipped} patients already recorded (skipped), "
              f"{len(missing)} to record", flush=True)
    for round_no in range(1, args.auto_resume_rounds + 1):
        if not missing:
            break
        print(f"[medboard] round {round_no}: {len(missing)} patients to record", flush=True)
        with ThreadPoolExecutor(max_workers=args.patient_workers) as ppool:
            futures = {
                ppool.submit(process_patient, pid, bank[pid], args.output_dir, args.model,
                             args.api_key, args.base_url, args.max_questions,
                             args.question_workers): pid
                for pid in missing
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                n_q, n_correct, n_skills = fut.result()
                with lock:
                    done += 1
                    total_q += n_q
                    total_correct += n_correct
                    total_skills += n_skills
                print(f"[medboard] ({done}/{len(pids)}) patient {pid}: {n_q} questions, "
                      f"{n_correct} correct, {n_skills} skills written", flush=True)
        missing = missing_pids()
        if missing:
            print(f"[medboard] {len(missing)} patients still missing, "
                  f"retrying in round {round_no + 1}", flush=True)

    print(f"\n[medboard] done: {total_q} questions, {total_correct} correct "
          f"({total_correct / max(total_q, 1) * 100:.1f}%), {total_skills} skills "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
