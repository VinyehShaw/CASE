#!/bin/bash
# Stage 1 - patient-level 40/50/10 split (md5-stable, seed 20260926) computed
# from the raw trajectory directories of the three roles:
#   Anatomist  (tool recordings originally labelled "vista3d")
#   Radiologist (originally "medpanel_v2")
#   Consultant  (originally "medboard")
# Bands: [0,40) SFT / [40,90) RL / [90,100) test.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" "$SNAP/scripts/split_dataset_603010.py" \
    --trajectory-dir "$TRAJ_ANATOMIST" \
    --trajectory-dir "$TRAJ_RADIOLOGIST" \
    --trajectory-dir "$TRAJ_CONSULTANT" \
    --sft-frac 0.4 --rl-frac 0.5 \
    --seed 20260926 --mpp 4
