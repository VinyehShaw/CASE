"""Mixed tool-calling SFT — trains on both vista3d and medpanel_v2 trajectories.

Each sample is tokenized with only its own tool schema (3 tools), determined
by inspecting tool_calls in the trajectory. This prevents the model from
learning to call cross-domain tools at inference time.

Does NOT modify run_anatomist_tool_sft.py — all changes are monkey-patches.

Usage:
    accelerate launch --config_file configs/accelerate_config/vista3d_sft.yaml \
        scripts/run_mixed_tool_sft.py \
        --trajectory-dir /path/to/vista3d_traj /path/to/medpanel_v2_traj \
        --cache-dir /tmp/mys/mixed_cache \
        --output outputs/checkpoints/mixed_sft \
        --batch-size 2 --grad-accum 4 --num-epochs 1
"""
import argparse
import json
import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)

import run_anatomist_tool_sft as _v
from run_anatomist_tool_sft import build_cache, train

VISTA3D_TOOLS = _v.VISTA3D_TOOLS

MEDPANEL_V2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_images",
            "description": "List all available imaging modalities for this patient. Returns modality names, shapes, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "View a specific axial slice from a modality at a given visit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string", "description": "Modality name (e.g. 'brain_t1', 'heart_short_axis', 'liver_ideal', 'dixon_8_W')"},
                    "visit": {"type": "string", "description": "'v1' for baseline or 'v2' for follow-up"},
                    "slice_idx": {"type": "integer", "description": "Slice index (use -1 for middle slice)"},
                },
                "required": ["modality", "visit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_visits",
            "description": "Compare the same slice between Visit 1 and Visit 2 side-by-side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string", "description": "Modality name (e.g. 'brain_t1', 'liver_ideal', 'dixon_8_W')"},
                    "slice_idx": {"type": "integer", "description": "Slice index (use -1 for middle slice)"},
                },
                "required": ["modality"],
            },
        },
    },
]

VISTA3D_TOOL_NAMES = {"vista3d_segment", "vista3d_compare_stations", "vista3d_compare_visits"}

_single_turn = False


def merge_assistant_messages(messages):
    """Merge consecutive assistant messages (no tool_calls) into one.

    Tool-call rounds (assistant with tool_calls) are kept separate.
    """
    merged = []
    for msg in messages:
        if msg["role"] == "assistant" and not msg.get("tool_calls"):
            if (merged and merged[-1]["role"] == "assistant"
                    and not merged[-1].get("tool_calls")):
                prev = merged[-1].get("content", "")
                cur = msg.get("content", "")
                if isinstance(prev, str) and isinstance(cur, str):
                    merged[-1]["content"] = prev + "\n\n" + cur
                elif isinstance(prev, list) and isinstance(cur, list):
                    merged[-1]["content"] = prev + cur
                else:
                    merged[-1]["content"] = str(prev) + "\n\n" + str(cur)
            else:
                merged.append(dict(msg))
        else:
            merged.append(dict(msg))
    return merged


def get_tools_for_sample(messages):
    """Detect which tool set a sample uses based on tool_calls."""
    for m in messages:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc["function"]["name"]
                if name in VISTA3D_TOOL_NAMES:
                    return VISTA3D_TOOLS
            return MEDPANEL_V2_TOOLS
    return VISTA3D_TOOLS


def tokenize_with_spans_mixed(sample, processor, max_length, resize=512):
    """Per-sample tool schema version of tokenize_with_spans."""
    msgs = _v.materialize_images(sample["messages"], resize)
    if _single_turn:
        msgs = merge_assistant_messages(msgs)
    tools = get_tools_for_sample(msgs)

    full = processor.apply_chat_template(
        msgs, tools=tools, tokenize=True, return_dict=True, return_tensors="pt",
    )
    input_ids = full["input_ids"][0]
    if input_ids.shape[0] > max_length:
        return None

    labels = [-100] * input_ids.shape[0]
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        start_out = processor.apply_chat_template(
            msgs[:i], tools=tools, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        end_out = processor.apply_chat_template(
            msgs[:i + 1], tools=tools, tokenize=True, return_dict=True, return_tensors="pt",
        )
        start = start_out["input_ids"].shape[1]
        end = end_out["input_ids"].shape[1]
        if end <= input_ids.shape[0] and start < end:
            labels[start:end] = input_ids[start:end].tolist()

    if all(l == -100 for l in labels):
        return None

    entry = {
        "input_ids": input_ids.tolist(),
        "labels": labels,
        "attention_mask": full["attention_mask"][0].tolist(),
    }
    if "pixel_values" in full:
        entry["pixel_values"] = full["pixel_values"]
    if "image_grid_thw" in full:
        entry["image_grid_thw"] = full["image_grid_thw"]
    return entry


_original_load_agent_dataset = _v.load_agent_dataset

import random as _random

_kept_sets = {}  # {trajectory_dir: set of (patient_id, question_idx)}
_max_samples_per_dir = None
_max_per_patient = 0  # 0 = no cap; else keep N questions per patient (all patients kept)


def load_agent_dataset_mixed(trajectory_dirs, only_correct=False):
    """Load agent_*.jsonl from one or more directories.

    If _kept_sets is populated (via --kept-manifest), each directory's samples
    are filtered to only those (patient_id, question_idx) pairs in the manifest.
    If _max_samples_per_dir is set, each directory is randomly subsampled.
    Each directory's samples are shuffled before concatenation for interleaving.
    """
    if isinstance(trajectory_dirs, str):
        trajectory_dirs = [trajectory_dirs]

    all_samples = []
    for d in trajectory_dirs:
        kept = _kept_sets.get(d)
        if kept is not None:
            # The loader globs every agent_*.jsonl and parses all of it before the
            # manifest filter below runs, so with an SFT-only manifest about half
            # the NAS reads are thrown away and the stage sits silent for minutes.
            # The filename carries the patient id, so enumerate only wanted files.
            import glob as _glob_mod
            wanted = {pid for pid, _ in kept}
            _orig_glob = _glob_mod.glob

            def _filtered_glob(pattern, *a, **kw):
                paths = _orig_glob(pattern, *a, **kw)
                if not pattern.endswith("agent_*.jsonl"):
                    return paths
                keep = [p for p in paths
                        if os.path.basename(p)[len("agent_"):-len(".jsonl")] in wanted]
                print(f"  {d}: reading {len(keep)}/{len(paths)} agent files "
                      f"(skipped {len(paths) - len(keep)} not in manifest)", flush=True)
                return keep

            _glob_mod.glob = _filtered_glob
            try:
                samples = _original_load_agent_dataset(d, only_correct)
            finally:
                _glob_mod.glob = _orig_glob
        else:
            samples = _original_load_agent_dataset(d, only_correct)
        if kept is not None:
            before = len(samples)
            samples = [s for s in samples
                       if (str(s["patient_id"]), s["question_idx"]) in kept]
            print(f"  {d}: {len(samples)}/{before} after manifest filter")
        else:
            print(f"  {d}: {len(samples)} samples")
        if _max_per_patient and _max_per_patient > 0:
            import hashlib
            by_pid = {}
            for s in samples:
                by_pid.setdefault(str(s["patient_id"]), []).append(s)
            capped = []
            for pid in sorted(by_pid):
                plist = by_pid[pid]
                if len(plist) > _max_per_patient:
                    rng = _random.Random(int(hashlib.md5(pid.encode()).hexdigest(), 16))
                    rng.shuffle(plist)
                    plist = plist[:_max_per_patient]
                capped.extend(plist)
            print(f"  {d}: per-patient cap {_max_per_patient}: "
                  f"{len(samples)} -> {len(capped)} ({len(by_pid)} patients, all kept)")
            samples = capped
        if _max_samples_per_dir is not None and len(samples) > _max_samples_per_dir:
            _random.shuffle(samples)
            samples = samples[:_max_samples_per_dir]
            print(f"  {d}: subsampled to {_max_samples_per_dir}")
        _random.shuffle(samples)
        all_samples.extend(samples)
    print(f"Loaded {len(all_samples)} total samples from {len(trajectory_dirs)} directories")
    return all_samples


# Monkey-patch: per-sample tools + multi-dir support
_v.tokenize_with_spans = tokenize_with_spans_mixed
_v.load_agent_dataset = load_agent_dataset_mixed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed Tool-Calling SFT (vista3d + medpanel_v2)")
    parser.add_argument("--trajectory-dir", type=str, nargs="+", required=True,
                        help="One or more directories containing agent_*.jsonl files")
    parser.add_argument("--model", type=str,
                        default="/mnt/faster_nas/benyou.mys/medixrl/checkpoints/sft_mm_v3/checkpoint-709")
    parser.add_argument("--output", type=str,
                        default="/mnt/faster_nas/benyou.mys/medixrl/checkpoints/mixed_sft")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=5,
                        help="keep only the N most recent checkpoint-*; -1 = keep ALL (default 5)")
    parser.add_argument("--resume", action="store_true",
                        help="(now default behavior) continue from the latest checkpoint-* in --output")
    parser.add_argument("--no-resume", action="store_true",
                        help="force training from --model base weights, ignoring existing checkpoints")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="checkpoint every N steps")
    parser.add_argument("--only-correct", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--kept-manifest", type=str, nargs="+", default=None,
                        help="Causal filter manifest JSONL (one per --trajectory-dir, same order). "
                             "Only (patient_id, question_idx) pairs in the manifest are loaded.")
    parser.add_argument("--max-samples-per-dir", type=int, default=None,
                        help="Randomly subsample at most N samples per trajectory directory.")
    parser.add_argument("--max-per-patient", type=int, default=0,
                        help="Keep at most N questions per patient (0 = no cap). Every patient "
                             "is retained; deterministic per-patient shuffle. Applied at load, "
                             "so building a cache with this set produces a per-patient-capped cache.")
    parser.add_argument("--local-cache-dir", type=str, default=None,
                        help="Copy subsampled cache files to local disk for faster I/O. "
                             "Cleaned up automatically after training.")
    parser.add_argument("--single-turn", type=str, default="false",
                        help="true=merge consecutive assistant text messages, false=keep multi-turn")
    args = parser.parse_args()

    traj_dirs = args.trajectory_dir if len(args.trajectory_dir) > 1 else args.trajectory_dir[0]

    _single_turn = args.single_turn.lower() == "true"
    print(f"Single-turn mode: {_single_turn}")

    if args.kept_manifest:
        if len(args.kept_manifest) != len(args.trajectory_dir):
            parser.error(f"--kept-manifest needs {len(args.trajectory_dir)} files "
                         f"(one per --trajectory-dir), got {len(args.kept_manifest)}")
        for traj_dir, manifest_path in zip(args.trajectory_dir, args.kept_manifest):
            kept = set()
            with open(manifest_path) as f:
                for line in f:
                    e = json.loads(line)
                    kept.add((str(e["patient_id"]), e["question_idx"]))
            _kept_sets[traj_dir] = kept
            print(f"Loaded {len(kept)} kept entries from {manifest_path} → {traj_dir}")

    if args.max_samples_per_dir is not None:
        _max_samples_per_dir = args.max_samples_per_dir
        print(f"Max samples per directory: {_max_samples_per_dir}")

    if args.max_per_patient and args.max_per_patient > 0:
        _max_per_patient = args.max_per_patient
        print(f"Per-patient question cap: {_max_per_patient}")

    if args.build_cache:
        if not args.cache_dir:
            parser.error("--build-cache requires --cache-dir")
        build_cache(
            trajectory_dir=traj_dirs,
            model_path=args.model,
            cache_dir=args.cache_dir,
            max_length=args.max_length,
            resize=args.resize,
            only_correct=args.only_correct,
            workers=args.cache_workers,
        )
        sys.exit(0)

    max_samples = None
    if args.max_samples_per_dir is not None:
        max_samples = args.max_samples_per_dir * len(args.trajectory_dir)

    train(
        trajectory_dir=traj_dirs,
        model_path=args.model,
        output_dir=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        resize=args.resize,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume=(not args.no_resume),
        only_correct=args.only_correct,
        cache_dir=args.cache_dir,
        cache_workers=args.cache_workers,
        max_samples=max_samples,
        local_cache_dir=args.local_cache_dir,
    )
