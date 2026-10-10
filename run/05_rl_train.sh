#!/bin/bash
# Stage 3b - GRPO + OPSD multi-turn tool RL (verl, FSDP actor + vLLM rollout).
#
#   Smoke test:  MODEL_PATH=<SFT ckpt> ROLLOUT_N=2 MAX_TURNS=4 bash run/05_rl_train.sh   (Ctrl-C after 1-2 steps)
#   Full run:    MODEL_PATH=<SFT ckpt>                    bash run/05_rl_train.sh
#
# Requires: OPENAI_API_KEY (DashScope-compatible) for the causal-judge reward,
# and the RL parquet + tool config produced by run/04_rl_prepare.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$SNAP"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=<dashscope key> first}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

# Escape hatches proven on the reference environment (see README >> Notes).
# Without them vLLM picks the FlashInfer sampler, whose JIT needs a full CUDA
# toolkit that the shipped conda env does not include.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export FLA_CACHE_RESULTS="${FLA_CACHE_RESULTS:-0}"
export FLA_USE_FAST_OPS="${FLA_USE_FAST_OPS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
# Let scripts/vllm_patch prefer THIS package's distill copy in every process.
export CASE_CODE_ROOT="$SNAP"
# Cache directories used by the Consultant tooling (report mock tool etc.).
export MEDPANEL_V2_TOOL_CACHE_DIR="${MEDPANEL_V2_TOOL_CACHE_DIR:-$DATA_ROOT/rl/tool_env/medpanel_v2_tool_cache}"
export VISTA3D_TOOL_CACHE_DIR="${VISTA3D_TOOL_CACHE_DIR:-$DATA_ROOT/rl/tool_env/vista3d_tool_cache}"
export MEDGEMMA_IMAGE_BASE="${MEDGEMMA_IMAGE_BASE:-$DATA_ROOT/rl/tool_env/medgemma_0820}"

PARQUET="${TRAIN_PARQUET:-$OUT_ROOT/rl/verl_train_tool.parquet}"
[ -f "$PARQUET" ] || { echo "ERROR: RL parquet not found: $PARQUET (run run/04_rl_prepare.sh first)"; exit 1; }
export TRAIN_FILES="['$PARQUET']"
export VAL_FILES="$PARQUET"

export TOOL_CONFIG="${TOOL_CONFIG:-$OUT_ROOT/generated/consultant_tool_config_case.yaml}"
[ -f "$TOOL_CONFIG" ] || { echo "ERROR: tool config not found: $TOOL_CONFIG (run run/04_rl_prepare.sh first)"; exit 1; }

export MODEL_PATH="${MODEL_PATH:-$OUT_ROOT/sft}"
[ -e "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH"; exit 1; }

bash "$SNAP/scripts/run_verl_qwen3vl_tool_rl.sh"
