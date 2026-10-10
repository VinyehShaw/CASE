#!/bin/bash
# Stage 4 - evaluation: build the TEST bank (0.1 split) and run the 3-module
# evaluation (Anatomist / Radiologist / Consultant) against a served checkpoint.
#
# Usage: CHECKPOINT=<model dir or SFT checkpoint> bash run/06_eval.sh
# Requires the eval assets under $DATA_ROOT/eval_assets (see README >> Evaluation).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Cache directories used by the role tooling (Consultant view_image etc.).
export MEDPANEL_V2_TOOL_CACHE_DIR="${MEDPANEL_V2_TOOL_CACHE_DIR:-$DATA_ROOT/rl/tool_env/medpanel_v2_tool_cache}"
export VISTA3D_TOOL_CACHE_DIR="${VISTA3D_TOOL_CACHE_DIR:-$DATA_ROOT/rl/tool_env/vista3d_tool_cache}"
export MEDGEMMA_IMAGE_BASE="${MEDGEMMA_IMAGE_BASE:-$DATA_ROOT/rl/tool_env/medgemma_0820}"
# Let scripts/vllm_patch anchor on THIS package's code in every eval process.
export CASE_CODE_ROOT="$SNAP"
BANK="$OUT_ROOT/eval_bank"
if [ ! -f "$BANK/verl_train_tool.parquet" ]; then
    "$PY" -m distill.opd_rl.verl_opsd.prepare_mixed_tool_rl_data --test-only \
        --medpanel-v2-dir "$TRAJ_RADIOLOGIST" \
        --vista3d-dir "$TRAJ_ANATOMIST" \
        --medboard-dir "$TRAJ_CONSULTANT" \
        --vista3d-cache "$DATA_ROOT/rl/tool_env/vista3d_tool_cache/tool_cache.json" \
        --patient-ids-file "$SPLIT_RADIOLOGIST/test_patients.txt" \
        --output-dir "$BANK" --train-ratio 1.0
fi
# Render the eval runner with local asset paths (README >> Evaluation explains
# every knob: PORT / TP_SIZE / MODEL_NAME / OUTPUT_BASE / VAL_PARQUET).
# Eval asset root (override for layouts that keep the assets elsewhere,
# e.g. EVAL_ASSETS=$DATA_ROOT/eval/env).
EVAL_ASSETS="${EVAL_ASSETS:-$DATA_ROOT/eval_assets}"
GEN="$OUT_ROOT/generated/run_mixed_eval_case.sh"
sed -e "s|/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2|$EVAL_ASSETS/ukb_dixon_vista3d_v2|g" \
    -e "s|/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d|$EVAL_ASSETS/ukb_dixon_vista3d|g" \
    -e "s|/mnt/faster_nas/benyou.mys/dataset/ukb_mri_all|$EVAL_ASSETS/ukb_mri_all|g" \
    -e "s|/mnt/faster_nas/benyou.mys/medixrl/data/medgemma_report_cache|$EVAL_ASSETS/medgemma_report_cache|g" \
    -e "s|/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store|$EVAL_ASSETS/medboard_skill_store|g" \
    -e "s|/mnt/faster_nas/benyou.mys/git/ezdistill|$SNAP|g" \
    -e "s|/opt/conda/envs/medix2|$(dirname "$(dirname "$PY")")|g" \
    "$SNAP/scripts/run_mixed_eval.sh" > "$GEN"
# The shipped env has no full CUDA toolkit (no cicc), so vLLM's torch.compile
# allreduce-rmsnorm fusion would JIT-fail during CUDA-graph capture; serve
# eagerly instead (same knob the RL rollout uses).
sed -i 's|^vllm serve "${CHECKPOINT}" \\$|vllm serve "${CHECKPOINT}" --enforce-eager \\|' "$GEN"
grep -q -- '--enforce-eager' "$GEN" || { echo "ERROR: --enforce-eager injection failed" >&2; exit 1; }
export VAL_PARQUET="${VAL_PARQUET:-$BANK/verl_train_tool.parquet}"
export OUTPUT_BASE="${OUTPUT_BASE:-$OUT_ROOT/eval}"
echo "checkpoint (MODEL): ${CHECKPOINT:?export CHECKPOINT=<model path> first}"
bash "$GEN"
