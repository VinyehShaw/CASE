#!/bin/bash
# Optional data-prep utility (background-safe, live log): causally filter the
# recorded Anatomist trajectories - keeps questions whose trajectory causally
# supports the recorded answer. Requires OPENAI_API_KEY (DashScope judge).
#
# Usage: OPENAI_API_KEY=<key> bash scripts/run_causal_filter_anatomist.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$CODE_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$CODE_ROOT/out}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=<dashscope key> first}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
PY="${PY:-/opt/conda/envs/case/bin/python}"

TRAJ="${TRAJ_ANATOMIST:-$DATA_ROOT/sft/trajectories/anatomist}"
OUT="${OUT_ROOT}/causal_filter"
mkdir -p "$OUT"
LOG="${OUT}/anatomist.log"
OUT_MANIFEST="${OUT}/anatomist_kept.jsonl"

PYTHONUNBUFFERED=1 nohup "$PY" "$CODE_ROOT/scripts/filter_trajectories_causal.py" \
    --trajectory-dir "$TRAJ" \
    --threshold 0.4 \
    --judge-model qwen3-max \
    --output-manifest "$OUT_MANIFEST" \
    > "$LOG" 2>&1 &
echo "started PID $!, log: $LOG, manifest: $OUT_MANIFEST"
echo "watch with: tail -f $LOG"
