#!/bin/bash
# Stage 2b - SFT training (reuses the tokenized cache from run/02).
#
# Requires: MODEL_PATH = path to the Qwen3-VL-8B-Instruct base model.
# Uses accelerate + DeepSpeed ZeRO-2 over NPROC GPUs (default 8).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$SNAP"
: "${MODEL_PATH:?export MODEL_PATH=<path to Qwen3-VL-8B-Instruct> first}"
"$(dirname "$PY")/accelerate" launch --config_file configs/accelerate_config/sft_8gpu_ds.yaml \
    --num_processes "${NPROC:-8}" scripts/run_mixed_tool_sft.py --model "$MODEL_PATH" \
    --trajectory-dir "$TRAJ_ANATOMIST" "$TRAJ_RADIOLOGIST" "$TRAJ_CONSULTANT" \
    --kept-manifest \
        "$SPLIT_ANATOMIST/sft_kept.jsonl" \
        "$SPLIT_RADIOLOGIST/sft_kept.jsonl" \
        "$SPLIT_CONSULTANT/sft_kept.jsonl" \
    --output "$OUT_ROOT/sft" --cache-dir "$OUT_ROOT/sft_cache" \
    --max-length 8192 --resize 512 --batch-size 2 --grad-accum 2 \
    --num-epochs "${EPOCHS:-1}"
