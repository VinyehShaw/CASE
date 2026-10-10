#!/bin/bash
# Stage 3a - build the RL training parquet from the 0.3 split, and generate the
# tool config with local (DATA_ROOT) cache paths.
#
# CLI flag names (--vista3d-dir / --medpanel-v2-dir / --medboard-dir) are the
# builder's historical option names; they map to the three roles:
#   --vista3d-dir    -> Anatomist   ($TRAJ_ANATOMIST)
#   --medpanel-v2-dir-> Radiologist ($TRAJ_RADIOLOGIST)
#   --medboard-dir   -> Consultant  ($TRAJ_CONSULTANT)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$SNAP"
"$PY" -m distill.opd_rl.verl_opsd.prepare_mixed_tool_rl_data \
    --medpanel-v2-dir "$TRAJ_RADIOLOGIST" \
    --vista3d-dir "$TRAJ_ANATOMIST" \
    --medboard-dir "$TRAJ_CONSULTANT" \
    --vista3d-cache "$DATA_ROOT/rl/tool_env/vista3d_tool_cache/tool_cache.json" \
    --patient-ids-file "$SPLIT_RADIOLOGIST/rl_patients.txt" \
    --question-manifest "$SPLIT_RADIOLOGIST/rl_kept.jsonl" \
    --output-dir "$OUT_ROOT/rl" --train-ratio 1.0
sed -e "s|/mnt/faster_nas/benyou.mys/medixrl/data/medpanel_v2_tool_cache|$DATA_ROOT/rl/tool_env/medpanel_v2_tool_cache|" \
    -e "s|/mnt/faster_nas/benyou.mys/medixrl/data/vista3d_tool_cache|$DATA_ROOT/rl/tool_env/vista3d_tool_cache|" \
    -e "s|/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store|$DATA_ROOT/rl/tool_env/medboard_skill_store|" \
    distill/tool/consultant_tool_config.yaml > "$OUT_ROOT/generated/consultant_tool_config_case.yaml"
echo "OK: $OUT_ROOT/rl/verl_train_tool.parquet + $OUT_ROOT/generated/consultant_tool_config_case.yaml"
