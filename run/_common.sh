# Shared header for the run/*.sh pipeline scripts (source it, do not execute).
#
# Resolves:
#   CODE_ROOT  - this code package (repository root)
#   SNAP       - alias of CODE_ROOT (stage scripts run from here)
#   PY         - Python interpreter of the conda environment (install via
#                run/00_env.sh, or use the published docker image; see README)
#   DATA_ROOT  - dataset root (override with the DATA_ROOT env var)
#   OUT_ROOT   - outputs root (override with the OUT_ROOT env var)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"
SNAP="$CODE_ROOT"

DATA_ROOT="${DATA_ROOT:-$CODE_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$CODE_ROOT/out}"
mkdir -p "$OUT_ROOT/generated"

# Per-role data directories. The released dataset (see README) follows this
# layout; override individually if your copy differs.
#   $DATA_ROOT/sft/trajectories/{anatomist,radiologist,consultant}
#   $DATA_ROOT/splits/{anatomist,radiologist,consultant}/split_<seed>/
TRAJ_ANATOMIST="${TRAJ_ANATOMIST:-$DATA_ROOT/sft/trajectories/anatomist}"
TRAJ_RADIOLOGIST="${TRAJ_RADIOLOGIST:-$DATA_ROOT/sft/trajectories/radiologist}"
TRAJ_CONSULTANT="${TRAJ_CONSULTANT:-$DATA_ROOT/sft/trajectories/consultant}"
SPLITS="${SPLITS:-$DATA_ROOT/splits}"
# Per-role split directories (override for legacy layouts, e.g.
# SPLIT_ANATOMIST=$DATA_ROOT/splits/vista3d/split_20260926).
SPLIT_ANATOMIST="${SPLIT_ANATOMIST:-$SPLITS/anatomist/split_20260926}"
SPLIT_RADIOLOGIST="${SPLIT_RADIOLOGIST:-$SPLITS/radiologist/split_20260926}"
SPLIT_CONSULTANT="${SPLIT_CONSULTANT:-$SPLITS/consultant/split_20260926}"

if [ -z "${PY:-}" ]; then
    for c in /opt/conda/envs/case/bin/python /opt/conda/envs/medix2/bin/python; do
        [ -x "$c" ] && PY="$c" && break
    done
fi
[ -x "${PY:-/nonexistent}" ] || {
    echo "ERROR: conda env not found. Install it with run/00_env.sh or use the docker image; see README."
    exit 1
}
export CODE_ROOT SNAP PY DATA_ROOT OUT_ROOT
export TRAJ_ANATOMIST TRAJ_RADIOLOGIST TRAJ_CONSULTANT SPLITS
export SPLIT_ANATOMIST SPLIT_RADIOLOGIST SPLIT_CONSULTANT
