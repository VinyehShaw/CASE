"""Prepare multi-turn tool RL data from trajectory dense_*.json files.

Builds parquet with tools_kwargs (no images column) for verl AgentLoop multi-turn RL.
The model calls tools during rollout to dynamically select and view images.

Output columns:
  data_source, prompt, reward_model, extra_info, agent_name

Usage:
  python -m distill.opd_rl.verl_opsd.prepare_mixed_tool_rl_data \
      --medpanel-v2-dir /mnt/faster_nas/benyou.mys/dataset/medpanel_v2_trajectory_0724 \
      --vista3d-dir /mnt/faster_nas/benyou.mys/dataset/vista3d_traj_0728_nw \
      --vista3d-cache /mnt/faster_nas/benyou.mys/medixrl/data/vista3d_tool_cache/tool_cache.json \
      --output-dir /mnt/faster_nas/benyou.mys/medixrl/data/rl_tool_data
"""

import json
import os
import random
import argparse

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# System prompts (hardcoded from tool files for self-containment)
# ============================================================

MEDPANEL_V2_SYSTEM_PROMPT = """You are MedPanel, an autonomous medical imaging analysis agent. You examine longitudinal medical imaging data (Visit 1 vs Visit 2) to answer clinical questions about disease progression, changes, and diagnosis.

## Available Tools
You have tools to examine patient imaging data:
1. **list_available_images** — List all available imaging modalities and their metadata
2. **view_image** — View a specific axial slice from a modality at a given visit (returns an image)
3. **compare_visits** — Side-by-side comparison of the same slice between Visit 1 and Visit 2

## Strategy
1. Call list_available_images first to see what's available
2. Select 1-2 most relevant modalities based on the clinical question
3. Use view_image or compare_visits to examine specific slices
4. Limit yourself to 4 tool calls total — be efficient!
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
- Clinical Insight: [synthesized analysis of the changes and their significance]

## Phase 5: Final Diagnosis
- Answer: [single letter A/B/C/D]
- Reasoning: [concise justification for the answer]
"""

VISTA3D_SYSTEM_PROMPT = """You are a medical imaging assistant analyzing whole-body MRI segmentation data.

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

## Phase 2: V1 Analysis
- Findings: <key volume data and observations from baseline>

## Phase 3: V2 Analysis & Comparison
- Findings: <key volume data from follow-up>
- Volume Change: <direction and magnitude of change>

## Phase 4: Insight Synthesis
- Clinical Insight: <synthesized analysis of what the volume changes mean>

## Phase 5: Final Diagnosis
Answer: <single letter A/B/C/D>
Reasoning: <concise justification>
"""

# Fallback only (normally imported from distill.tool.consultant.consultant_tool)
MEDBOARD_SYSTEM_PROMPT = """You are MedBoard, the FINAL ARBITRATION agent. Integrate ALL evidence sources (imaging, volumetry, MedGemma reports, skill library) and beat any single one. Required workflow: recall_skills FIRST, then multi-source review, actively DOUBT key claims with [VERIFIED]/[SUSPICIOUS]/[CONTRADICTED] labels, then a final message with ALL five headers: ## Phase 1: Skill Recall & Review Plan / ## Phase 2: Multi-source Review / ## Phase 3: Doubt & Verification / ## Phase 4: Reflective Insight / ## Phase 5: Final Answer (Answer: <letter(s)>). Do NOT call write_skill in this stage."""


# ============================================================
# Tool name sets
# ============================================================

MEDPANEL_V2_TOOLS = ["list_available_images", "view_image", "compare_visits"]
VISTA3D_TOOLS = ["vista3d_segment", "vista3d_compare_stations", "vista3d_compare_visits"]
MEDBOARD_TOOLS = (
    ["list_available_images", "view_image", "compare_visits"]
    + VISTA3D_TOOLS
    + ["list_available_reports", "get_report", "recall_skills", "write_skill"]
)


def _medboard_system_prompt():
    """Import the recording-side SYSTEM_PROMPT so RL prompts match the SFT
    distribution exactly (single source of truth)."""
    try:
        from distill.tool.consultant.consultant_tool import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    except Exception:
        return None


def build_tools_kwargs(patient_id, tool_names):
    """Build tools_kwargs dict for extra_info."""
    return {
        name: {"create_kwargs": {"patient_id": str(patient_id)}}
        for name in tool_names
    }


def build_medpanel_v2_prompt(patient_id, patient_info, question, options):
    """Build system + user messages for medpanel_v2."""
    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    user_content = (
        f"Patient ID: {patient_id}\n\n"
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Examine the available imaging data to answer this question. "
        f"Use at most 4 tool calls. "
        f"Be efficient — select the most relevant modality and examine key slices."
    )
    return [
        {"role": "system", "content": MEDPANEL_V2_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_vista3d_prompt(patient_id, patient_info, question, options, available_organs=None):
    """Build system + user messages for vista3d."""
    opt_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    organs_line = ""
    if available_organs:
        organs_line = f"Available organs in this patient's segmentation: {', '.join(available_organs[:20])}\n\n"
    user_content = (
        f"Patient ID: {patient_id}\n"
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_text}\n\n"
        f"{organs_line}"
        f"Instructions:\n"
        f"1. Identify the 1-3 organs most relevant to the question.\n"
        f"2. Call vista3d_compare_visits ONLY for those organs.\n"
        f"3. Analyze the volume change data AND overlay images, then write your 5-phase FINAL response and STOP.\n"
        f"   Do NOT call more than 3 tools total."
    )
    return [
        {"role": "system", "content": VISTA3D_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_medboard_prompt(patient_id, patient_info, question, options):
    """Build system + user messages for medboard (6-phase arbitration agent).
    The system prompt is imported from the recording-side module so RL prompts
    stay distribution-identical to the SFT trajectories."""
    sys_prompt = _medboard_system_prompt() or MEDBOARD_SYSTEM_PROMPT
    opt_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    user_content = (
        f"Patient ID: {patient_id}\n\n"
        f"Patient Info: {patient_info[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_text}\n\n"
        f"Work through the evidence sources with tool calls (recall_skills first), "
        f"doubt and verify key claims, then write your complete 6-phase final response."
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


def load_vista3d_organs(cache_path):
    """Load available organs per patient from vista3d tool cache."""
    organs_by_pid = {}
    if not cache_path or not os.path.exists(cache_path):
        return organs_by_pid
    with open(cache_path) as f:
        cache = json.load(f)
    for key in cache:
        parts = key.split("::")
        if len(parts) == 3:
            pid, tool, organ = parts
            if tool == "vista3d_segment":
                organs_by_pid.setdefault(pid, []).append(organ)
    return organs_by_pid


def _in_split(pid, split, rl_frac, split_seed):
    """Same per-patient hash bucket as prepare_fixed_img_rl_data.py, so tool and
    fixed-image datasets select an identical set of RL patients."""
    if split not in ("sft", "rl"):
        return True
    import hashlib
    h = int(hashlib.md5(f"{split_seed}:{pid}".encode()).hexdigest(), 16)
    is_rl = h % 100 < int(rl_frac * 100)
    return is_rl if split == "rl" else (not is_rl)


def _is_test(pid, test_frac, test_seed):
    """Patient-level held-out TEST bucket (independent salt, disjoint from the
    SFT/RL pool). test_frac<=0 disables it. TEST patients are excluded from
    both SFT and RL training and are reserved for final evaluation only."""
    if not test_frac or test_frac <= 0:
        return False
    import hashlib
    h = int(hashlib.md5(f"test:{test_seed}:{pid}".encode()).hexdigest(), 16)
    return h % 100 < int(test_frac * 100)


def _in_train_pool(pid, patient_frac, patient_seed):
    """Patient-level sampling over the non-TEST pool. Independent "train:" salt
    (disjoint from TEST). patient_frac>=1.0 keeps everyone. Uses the SAME salt
    and formula as scripts/gen_sft_kept_manifest.py, so passing the same
    patient_frac+patient_seed to both makes SFT and RL pick an identical
    patient list."""
    if patient_frac is None or patient_frac >= 1.0:
        return True
    import hashlib
    h = int(hashlib.md5(f"train:{patient_seed}:{pid}".encode()).hexdigest(), 16)
    return h % 100 < int(patient_frac * 100)


def process_dataset(traj_dir, data_source, system_prompt_type, organs_by_pid=None, max_samples=None,
                    split="all", rl_frac=0.5, split_seed=20260805, max_per_patient=0, seed=0,
                    test_frac=0.0, test_seed=20260812, patient_frac=1.0, patient_seed=20260812,
                    test_only=False, patient_ids=None, question_manifest=None):
    """Process one trajectory directory into parquet rows."""
    dense_files = sorted(
        f for f in os.listdir(traj_dir)
        if f.startswith("dense_") and f.endswith(".json")
    )
    dense_files = [f for f in dense_files if _in_split(f[len("dense_"):-len(".json")], split, rl_frac, split_seed)]
    n_before_test = len(dense_files)
    if patient_ids is not None:
        dense_files = [f for f in dense_files
                       if f[len("dense_"):-len(".json")] in patient_ids]
        print(f"  {traj_dir}: {len(dense_files)} dense files (patient-ids-file override)")
    elif test_only:
        dense_files = [f for f in dense_files
                       if _is_test(f[len("dense_"):-len(".json")], test_frac, test_seed)]
        print(f"  {traj_dir}: {len(dense_files)} TEST patients kept "
              f"(test_only, from {n_before_test})")
    else:
        dense_files = [f for f in dense_files
                       if not _is_test(f[len("dense_"):-len(".json")], test_frac, test_seed)]
        n_before_frac = len(dense_files)
        dense_files = [f for f in dense_files
                       if _in_train_pool(f[len("dense_"):-len(".json")], patient_frac, patient_seed)]
        print(f"  {traj_dir}: {len(dense_files)} dense files (split={split}, "
              f"excluded {n_before_test - len(dense_files)} TEST, "
              f"{n_before_frac - len(dense_files)} by patient-frac)")

    rows = []
    for fname in dense_files:
        pid = fname[len("dense_"):-len(".json")]
        with open(os.path.join(traj_dir, fname)) as f:
            data = json.load(f)

        patient_id = str(data.get("patient_id", pid))
        patient_info = data.get("patient_info", "")
        questions = data.get("questions", [])
        if max_per_patient and len(questions) > max_per_patient:
            questions = list(questions)
            random.Random(seed + hash(pid) % 100000).shuffle(questions)
            questions = questions[:max_per_patient]
        if question_manifest is not None:
            questions = [q for q in questions
                         if (patient_id, q.get("question_idx", q.get("index", 0)))
                         in question_manifest]

        for q in questions:
            q_idx = q.get("question_idx", q.get("index", 0))
            question = q.get("question", "")
            options = q.get("options", {})
            ground_truth = str(q.get("ground_truth", "")).strip()
            consensus_insight = ""
            traj = q.get("trajectory", {})
            if isinstance(traj, dict):
                fr = traj.get("final_result", {})
                if isinstance(fr, dict):
                    consensus_insight = fr.get("consensus_insight", "")

            if system_prompt_type == "medpanel_v2":
                prompt = build_medpanel_v2_prompt(patient_id, patient_info, question, options)
                tool_names = MEDPANEL_V2_TOOLS
            elif system_prompt_type == "medboard":
                prompt = build_medboard_prompt(patient_id, patient_info, question, options)
                tool_names = MEDBOARD_TOOLS
            else:
                available_organs = organs_by_pid.get(pid, []) if organs_by_pid else None
                prompt = build_vista3d_prompt(patient_id, patient_info, question, options, available_organs)
                tool_names = VISTA3D_TOOLS

            tools_kwargs = build_tools_kwargs(patient_id, tool_names)

            rows.append({
                "data_source": data_source,
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "reward_model": json.dumps({"ground_truth": ground_truth}),
                "extra_info": json.dumps({
                    "tools_kwargs": tools_kwargs,
                    "ground_truth": ground_truth,
                    "question": question,
                    "options": options,
                    "patient_id": patient_id,
                    "question_idx": q_idx,
                    "teacher_insight": consensus_insight,
                }, ensure_ascii=False),
                "agent_name": "tool_agent",
            })

            if max_samples and len(rows) >= max_samples:
                break
        if max_samples and len(rows) >= max_samples:
            break

    print(f"  {data_source}: {len(rows)} samples")
    return rows


def write_parquet(rows, output_path):
    """Write rows to parquet file."""
    table = pa.table({
        "data_source": [r["data_source"] for r in rows],
        "prompt": [r["prompt"] for r in rows],
        "reward_model": [r["reward_model"] for r in rows],
        "extra_info": [r["extra_info"] for r in rows],
        "agent_name": [r["agent_name"] for r in rows],
    })
    pq.write_table(table, output_path)
    print(f"  {len(rows)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-turn tool RL data")
    parser.add_argument("--medpanel-v2-dir", type=str, required=True)
    parser.add_argument("--vista3d-dir", type=str, required=True)
    parser.add_argument("--medboard-dir", type=str, default=None,
                        help="optional medboard trajectory dir; adds medboard_opsd rows "
                             "(6-phase prompt + 10 tools) to the RL mix")
    parser.add_argument("--vista3d-cache", type=str, default=None,
                        help="Path to vista3d tool_cache.json for organ lookup")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--patient-ids-file", type=str, default=None,
                        help="optional patient-id list (one per line); when set it "
                             "replaces the hash-band/test filters, e.g. the output of "
                             "scripts/split_dataset_603010.py")
    parser.add_argument("--question-manifest", type=str, default=None,
                        help="optional jsonl of {patient_id, question_idx}; keeps only "
                             "those questions (e.g. split_*/rl_kept.jsonl)")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-dir", type=int, default=None)
    parser.add_argument("--split", choices=["all", "sft", "rl"], default="all",
                        help="patient-level split; same hash bucket as prepare_fixed_img_rl_data.py")
    parser.add_argument("--rl-frac", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=20260805)
    parser.add_argument("--max-per-patient", type=int, default=0, help="0 = no cap")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="patient-level held-out TEST fraction (independent hash salt); "
                             "0 = disable. TEST patients are excluded from SFT/RL and reserved for eval.")
    parser.add_argument("--test-seed", type=int, default=20260812)
    parser.add_argument("--patient-frac", type=float, default=1.0,
                        help="keep this fraction of non-TEST patients for training (1.0 = all). "
                             "Patient-level, seeded; same value as gen_sft_kept_manifest.py makes "
                             "SFT and RL use the identical patient list.")
    parser.add_argument("--patient-seed", type=int, default=20260812)
    parser.add_argument("--test-only", action="store_true",
                        help="build the held-out TEST eval bank instead of the training pool: "
                             "keep ONLY TEST patients with ALL their questions "
                             "(forces split=all, patient_frac=1.0, max_per_patient=0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading vista3d organs from cache...")
    organs_by_pid = load_vista3d_organs(args.vista3d_cache)
    print(f"  {len(organs_by_pid)} patients with organ info")

    if args.test_only:
        args.split = "all"
        args.patient_frac = 1.0
        args.max_per_patient = 0
        args.train_ratio = 1.0

    pid_set = None
    if args.patient_ids_file:
        pid_set = {ln.strip() for ln in open(args.patient_ids_file) if ln.strip()}
    qman = None
    if args.question_manifest:
        qman = set()
        for ln in open(args.question_manifest):
            o = json.loads(ln)
            qman.add((str(o["patient_id"]), o["question_idx"]))

    _split_kw = dict(split=args.split, rl_frac=args.rl_frac, split_seed=args.split_seed,
                     max_per_patient=args.max_per_patient, seed=args.seed,
                     test_frac=args.test_frac, test_seed=args.test_seed,
                     patient_frac=args.patient_frac, patient_seed=args.patient_seed,
                     test_only=args.test_only, patient_ids=pid_set,
                     question_manifest=qman)

    # Collect the held-out TEST patient ids (for building the eval bank later).
    if args.test_frac and args.test_frac > 0:
        test_pids = set()
        for d in (args.medpanel_v2_dir, args.vista3d_dir):
            for f in os.listdir(d):
                if f.startswith("dense_") and f.endswith(".json"):
                    pid = f[len("dense_"):-len(".json")]
                    if _is_test(pid, args.test_frac, args.test_seed):
                        test_pids.add(pid)
        test_pid_path = os.path.join(args.output_dir, "test_patient_ids.txt")
        with open(test_pid_path, "w") as fh:
            for pid in sorted(test_pids):
                fh.write(pid + "\n")
        print(f"TEST bucket: {len(test_pids)} patients -> {test_pid_path}")

    print("\nProcessing medpanel_v2 trajectories...")
    mp_rows = process_dataset(
        args.medpanel_v2_dir, "medpanel_v2_opsd", "medpanel_v2",
        max_samples=args.max_samples_per_dir, **_split_kw,
    )

    print("\nProcessing vista3d trajectories...")
    vt_rows = process_dataset(
        args.vista3d_dir, "vista3d_opsd", "vista3d",
        organs_by_pid=organs_by_pid,
        max_samples=args.max_samples_per_dir, **_split_kw,
    )

    mb_rows = []
    if args.medboard_dir:
        print("\nProcessing medboard trajectories...")
        mb_rows = process_dataset(
            args.medboard_dir, "medboard_opsd", "medboard",
            max_samples=args.max_samples_per_dir, **_split_kw,
        )

    all_rows = mp_rows + vt_rows + mb_rows
    random.seed(args.seed)
    random.shuffle(all_rows)
    if args.train_ratio >= 1.0:
        splits = {"train": all_rows}
    else:
        split_idx = int(len(all_rows) * args.train_ratio)
        splits = {"train": all_rows[:split_idx], "eval": all_rows[split_idx:]}

    print(f"\nSplit: " + ", ".join(f"{len(v)} {k}" for k, v in splits.items()))
    for split_name, split_rows in splits.items():
        output_path = os.path.join(args.output_dir, f"verl_{split_name}_tool.parquet")
        write_parquet(split_rows, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
