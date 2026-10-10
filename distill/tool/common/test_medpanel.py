#!/usr/bin/env python3
"""
test_medpanel.py — MedPanelTool test script

Supports reading patient data from a parquet file and executing a multi-expert joint diagnostic discussion.
By default, processes all problems of one patient in a batch (one batch).

Usage:
    # import test (does not run the model)
    python test_medpanel.py --parquet train.parquet --patient-id 1007146 --import-only

    # batch: run all problems of one patient (default)
    export OPENAI_API_KEY="sk-xxx"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    python test_medpanel.py --parquet train.parquet --patient-id 1007146

    # single problem
    python test_medpanel.py --parquet train.parquet --patient-id 1007146 --question-idx 0

    # specify model
    python test_medpanel.py --parquet train.parquet --patient-id 1007146 --gpt-model qwen3.7-plus
"""

import argparse
import json
import os
import re
import sys
import traceback

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd


def load_patient_data(parquet_path: str, patient_id: str):
    """
    Load the data of the specified patient from a parquet file.

    Images are automatically grouped by modality+visit+type:
    images = {
        "Brain T1 MRI": {
            "visit_1": {
                "images": [bytes, ...],
                "types": ["montage", "representative slice", ...]
            },
            "visit_2": {
                "images": [bytes, ...],
                "types": ["montage", "representative slice", ...]
            }
        },
        ...
    }

    Returns:
        {
            "patient_info": str,
            "images": {modality: {"visit_1": {...}, "visit_2": {...}}},
            "problems": [str, ...],
            "solutions": [str, ...],
        }
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    target_id = str(patient_id)

    # First locate the row group containing the target patient (read only the patient_id column to avoid reading nested columns)
    found_rg = -1
    for rg in range(pf.num_row_groups):
        ids = pf.read_row_group(rg, columns=["patient_id"])["patient_id"].to_pylist()
        if target_id in [str(x) for x in ids]:
            found_rg = rg
            break

    if found_rg < 0:
        ids = pf.read_row_group(0, columns=["patient_id"])["patient_id"].to_pylist()[:10]
        raise ValueError(f"Patient {patient_id} not found. Available: {ids}...")

    # Read row by row with iter_batches (avoid the chunked array conversion bug for the nested list<struct<binary>> type)
    row = {}
    for batch in pf.iter_batches(batch_size=1, row_groups=[found_rg]):
        if str(batch["patient_id"][0].as_py()) == target_id:
            for col in batch.column_names:
                try:
                    row[col] = batch[col][0].as_py()
                except Exception as e:
                    print(f"  [Warning] failed to read column {col}: {e}")
                    row[col] = None
            break

    # Extract patient info
    patient_info = row.get("narrative_report", "") or ""

    # Extract images (grouped by modality + visit + type)
    images = {}
    for col, visit_key in [("visit_1_images", "visit_1"), ("visit_2_images", "visit_2")]:
        img_list = row.get(col)
        type_list = row.get(col.replace("_images", "_image_types"))

        if img_list is None or type_list is None:
            continue

        for i, (img, img_type) in enumerate(zip(img_list, type_list)):
            # Parse modality and type
            img_type_str = str(img_type)
            parts = img_type_str.split("/")

            if len(parts) < 3:
                continue

            modality = parts[0].strip()
            series = parts[1].strip()
            view_type = parts[2].strip()

            if modality not in images:
                images[modality] = {
                    "visit_1": {"images": [], "types": []},
                    "visit_2": {"images": [], "types": []}
                }

            # The image may be bytes or dict
            if isinstance(img, dict) and "bytes" in img:
                img_bytes = img["bytes"]
            elif isinstance(img, bytes):
                img_bytes = img
            else:
                continue

            # Record the image type (including series and view_type)
            full_type = f"{series}/{view_type}"
            images[modality][visit_key]["images"].append(img_bytes)
            images[modality][visit_key]["types"].append(full_type)

    # Extract problems and solutions
    problems = row.get("problems", [])
    solutions = row.get("solutions", [])

    if isinstance(problems, np.ndarray):
        problems = problems.tolist()
    if isinstance(solutions, np.ndarray):
        solutions = solutions.tolist()
    if problems is None:
        problems = []
    if solutions is None:
        solutions = []

    return {
        "patient_info": patient_info,
        "images": images,
        "problems": problems,
        "solutions": solutions,
    }


def parse_question(problem_text: str):
    """
    Parse question and options from the problem text.

    Returns:
        (question: str, options: dict)
    """
    # Extract the [Question] section
    question = ""
    if "[Question]" in problem_text:
        q_start = problem_text.index("[Question]") + len("[Question]")
        q_text = problem_text[q_start:].strip()
        if "Options:" in q_text:
            question = q_text[:q_text.index("Options:")].strip()
        else:
            question = q_text.split("\n")[0].strip()

    # Extract Options
    options = {}
    if "Options:" in problem_text:
        opt_start = problem_text.index("Options:") + len("Options:")
        opt_text = problem_text[opt_start:].strip()
        # Truncate non-option text
        if "Clinical reasoning" in opt_text:
            opt_text = opt_text[:opt_text.index("Clinical reasoning")].strip()

        for line in opt_text.split("\n"):
            line = line.strip()
            match = re.match(r"([A-Z])\.\s*(.+)", line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                options[key] = value

    return question, options


def test_import():
    """Test import"""
    print("Testing import of MedPanelTool...")
    from distill.tool.medpanel import MedPanelTool, PanelTrajectory
    print("  MedPanelTool ✓")
    print("  PanelTrajectory ✓")

    # Test tool schema
    tool = MedPanelTool.__new__(MedPanelTool)
    tool.TOOL_NAME = "medpanel_discuss"
    print(f"  tool_name: {tool.TOOL_NAME} ✓")

    # List supported expert roles
    from distill.tool.common.medpanel_tool import EXPERT_ROLES
    print(f"\n  Supported imaging modalities ({len(EXPERT_ROLES)}):")
    for mod, role in EXPERT_ROLES.items():
        print(f"    {mod} → {role['name']} ({role['cn_name']})")

    print("\nImport successful ✓")


def test_diagnose(args):
    """Test the full diagnostic workflow (supports concurrency)"""
    from distill.tool.medpanel import MedPanelTool

    # Load data
    print(f"\nLoading patient data: {args.patient_id}")
    data = load_patient_data(args.parquet, args.patient_id)

    print(f"  Imaging modalities:")
    for mod, visits in data["images"].items():
        v1_imgs = visits.get("visit_1", {}).get("images", [])
        v2_imgs = visits.get("visit_2", {}).get("images", [])
        v1_types = visits.get("visit_1", {}).get("types", [])
        v2_types = visits.get("visit_2", {}).get("types", [])

        v1_size = sum(len(b) for b in v1_imgs)
        v2_size = sum(len(b) for b in v2_imgs)

        print(f"    {mod}:")
        print(f"      Visit 1: {len(v1_imgs)} images ({v1_size/1024/1024:.1f} MB)")
        if v1_types:
            type_counts = {}
            for t in v1_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"        Types: {type_counts}")

        print(f"      Visit 2: {len(v2_imgs)} images ({v2_size/1024/1024:.1f} MB)")
        if v2_types:
            type_counts = {}
            for t in v2_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"        Types: {type_counts}")

    print(f"  Number of problems: {len(data['problems'])}")

    # Determine which problems to run
    if args.question_idx is not None:
        question_indices = [args.question_idx]
    else:
        # Default: run all problems
        question_indices = list(range(len(data["problems"])))

    # Initialize the tool
    print(f"\nInitializing MedPanelTool...")
    tool = MedPanelTool(
        model=args.gpt_model,
        output_dir=args.output_dir,
        max_api_concurrent=args.max_api_concurrent,
    )
    print(f"  Model: {args.gpt_model}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of problems to run this time: {len(question_indices)}")
    print(f"  Concurrent threads: {args.max_workers}")
    print(f"  API concurrency limit: {args.max_api_concurrent}")

    # Prepare batch data
    questions = []
    options_list = []
    solutions = []
    valid_indices = []

    for idx in question_indices:
        if idx >= len(data["problems"]):
            print(f"\nSkipping problem {idx}: out of range ({len(data['problems'])} problems in total)")
            continue

        problem_text = data["problems"][idx]
        solution = data["solutions"][idx] if idx < len(data["solutions"]) else None

        question, options = parse_question(problem_text)

        if not question:
            print(f"\nSkipping problem {idx}: failed to parse the question")
            continue

        questions.append(question)
        options_list.append(options)
        solutions.append(solution)
        valid_indices.append(idx)

    if not questions:
        print("\nNo valid problems, exiting")
        return

    # Run concurrently using diagnose_batch
    print(f"\n{'='*60}")
    print(f"Starting concurrent execution of {len(questions)} problems...")
    print(f"{'='*60}\n")

    batch_results = tool.diagnose_batch(
        patient_info=data["patient_info"],
        images=data["images"],
        questions=questions,
        options_list=options_list,
        solutions=solutions,
        patient_id=args.patient_id,
        verbose=True,
        max_workers=args.max_workers,
    )

    # Save results
    results = []
    for i, (idx, result) in enumerate(zip(valid_indices, batch_results)):
        result_with_meta = {
            "index": idx,
            "question": questions[i],
            "options": options_list[i],
            "ground_truth": solutions[i],
            "answer": result.get("answer"),
            "correct": result.get("correct"),
            "consensus_insight": result.get("consensus_insight"),
            "reasoning": result.get("reasoning"),
            "trajectory": result.get("trajectory"),
        }
        results.append(result_with_meta)

    # Store training data per patient (OPD format)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dense feedback: full trajectory JSON
    dense_path = os.path.join(args.output_dir, f"dense_{args.patient_id}.json")
    dense_data = {
        "patient_id": args.patient_id,
        "patient_info": data["patient_info"],
        "questions": [],
    }

    # 2. Sparse reward: SFT JSONL (one insight trajectory per problem per line)
    sparse_path = os.path.join(args.output_dir, f"sparse_{args.patient_id}.jsonl")

    for i, (idx, result) in enumerate(zip(valid_indices, batch_results)):
        trajectory = result.get("trajectory")

        if trajectory:
            # Dense feedback: full trajectory
            question_data = {
                "index": idx,
                "question": questions[i],
                "options": options_list[i],
                "ground_truth": solutions[i],
                "answer": result.get("answer"),
                "correct": result.get("correct"),
                "trajectory": trajectory.to_dict(),
            }
            dense_data["questions"].append(question_data)

            # Sparse reward: SFT format
            sft_messages = trajectory.to_sft_messages()
            with open(sparse_path, "a", encoding="utf-8") as f:
                json.dump(sft_messages, f, ensure_ascii=False)
                f.write("\n")

    # Save dense feedback
    with open(dense_path, "w", encoding="utf-8") as f:
        json.dump(dense_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Storage complete")
    print(f"{'='*60}")
    print(f"  Dense feedback (full trajectory): {dense_path}")
    print(f"  Sparse reward (SFT JSONL): {sparse_path}")
    print(f"  Total problems: {len(results)}")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"Diagnosis summary")
        print(f"{'='*60}")
        correct = sum(1 for r in results if r.get("correct"))
        total = len(results)
        print(f"  Total problems: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Total LLM calls: {tool.stats['total_llm_calls']}")

        # Save batch summary (used for OPD analysis)
        batch_path = os.path.join(args.output_dir, f"batch_{args.patient_id}.json")
        batch_data = [{
            "index": r["index"],
            "question": r["question"],
            "options": r["options"],
            "ground_truth": r["ground_truth"],
            "answer": r["answer"],
            "correct": r["correct"],
            "consensus_insight": r["consensus_insight"],
        } for r in results]
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        print(f"  Batch summary saved: {batch_path}")


def main():
    parser = argparse.ArgumentParser(description="MedPanel multi-expert joint diagnosis test")

    parser.add_argument("--parquet", type=str, help="Parquet data file path")
    parser.add_argument("--patient-id", type=str, help="Patient ID")
    parser.add_argument("--question-idx", type=int, default=None, help="Problem index (0-based); run all problems if not specified")
    parser.add_argument("--gpt-model", type=str, default="qwen3.7-plus", help="LLM model name")
    parser.add_argument("--output-dir", type=str, default="./medpanel_output", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent threads (default 3)")
    parser.add_argument("--max-api-concurrent", type=int, default=10, help="Global API concurrency limit (default 10)")
    parser.add_argument("--import-only", action="store_true", help="Test import only")

    args = parser.parse_args()

    if args.import_only:
        test_import()
        return

    # Non import-only mode requires parquet and patient-id
    if not args.parquet or not args.patient_id:
        parser.error("--parquet and --patient-id are required (unless --import-only)")

    test_diagnose(args)


if __name__ == "__main__":
    main()
