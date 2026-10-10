#!/bin/bash
# Stage 2a - build the tokenized SFT cache (assistant-only supervision).
# Re-running with a valid cache is a no-op ("Cache valid - skipping build").
#
# Requires: MODEL_PATH = path to the Qwen3-VL-8B-Instruct base model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$SNAP"
: "${MODEL_PATH:?export MODEL_PATH=<path to Qwen3-VL-8B-Instruct> first}"
"$PY" scripts/run_mixed_tool_sft.py --model "$MODEL_PATH" \
    --trajectory-dir "$TRAJ_ANATOMIST" "$TRAJ_RADIOLOGIST" "$TRAJ_CONSULTANT" \
    --kept-manifest \
        "$SPLIT_ANATOMIST/sft_kept.jsonl" \
        "$SPLIT_RADIOLOGIST/sft_kept.jsonl" \
        "$SPLIT_CONSULTANT/sft_kept.jsonl" \
    --output "$OUT_ROOT/sft" --cache-dir "$OUT_ROOT/sft_cache" \
    --build-cache --cache-workers "${CACHE_WORKERS:-64}" \
    --max-length 8192 --resize 512 --batch-size 2 --grad-accum 2
