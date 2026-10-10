#!/bin/bash
# Install the EXACT proven environment (bit-identical to the conda env that
# produced all published runs) from a shipped tar.
#
# Supports both packages:
#   case_env_0904.tar       (env already renamed to "case" inside)
#   medix2_env_0904.tar     (original name "medix2" inside; renamed on install if ENV_NAME set)
#
# Usage:
#   bash install_env_tar.sh [tar_path] [prefix]
#     tar_path default: <this dir>/case_env_0904.tar (falls back to medix2_env_0904.tar)
#     prefix   default: /opt/conda/envs
#   ENV_NAME=<name>    override the on-disk env name (default: whatever the tar contains)
#   COMPAT_LINK=0      do NOT create the <prefix>/medix2 -> <env> symlink (default: create it
#                      when the env name is not "medix2", so legacy hardcoded scripts still work)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAR="${1:-}"
if [ -z "$TAR" ]; then
    if [ -f "$HERE/case_env_0904.tar" ]; then TAR="$HERE/case_env_0904.tar"
    else TAR="$HERE/medix2_env_0904.tar"; fi
fi
PREFIX="${2:-/opt/conda/envs}"

[ -f "$TAR" ] || { echo "ERROR: tar not found: $TAR"; exit 1; }

INNER="$(tar -tf "$TAR" | head -1 | cut -d/ -f1)"
[ -n "$INNER" ] || { echo "ERROR: cannot read top-level dir from $TAR"; exit 1; }
NAME="${ENV_NAME:-$INNER}"
ENV="$PREFIX/$NAME"

if [ -x "$ENV/bin/python" ]; then
    echo "ERROR: $ENV already exists. Remove it first or pass a different prefix/ENV_NAME."
    exit 1
fi
mkdir -p "$PREFIX"
echo "extracting $(du -h "$TAR" | cut -f1) (inner name: $INNER) -> $PREFIX ..."
tar -xf "$TAR" -C "$PREFIX"

if [ "$INNER" != "$NAME" ]; then
    echo "renaming $INNER -> $NAME and fixing bin/ shebangs ..."
    mv "$PREFIX/$INNER" "$ENV"
    grep -l "#!/opt/conda/envs/$INNER/bin" "$ENV"/bin/* 2>/dev/null | \
        xargs -r sed -i "1s|/opt/conda/envs/$INNER/bin|/opt/conda/envs/$NAME/bin|"
fi
if [ "$NAME" != "medix2" ] && [ "${COMPAT_LINK:-1}" = "1" ] && [ ! -e "$PREFIX/medix2" ]; then
    ln -s "$ENV" "$PREFIX/medix2"
    echo "compat symlink: $PREFIX/medix2 -> $ENV"
fi

# sanity: the MEDIXRL-GUARD rope fix must be present in the installed verl
AL="$ENV/lib/python3.10/site-packages/verl/experimental/agent_loop/agent_loop.py"
grep -q "MEDIXRL-GUARD" "$AL" && echo "verl MEDIXRL-GUARD: OK" || echo "WARN: MEDIXRL-GUARD missing in $AL"

"$ENV/bin/python" - <<'PY'
import torch, transformers, ray, vllm, verl
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| ray", ray.__version__, "| vllm", vllm.__version__)
print("verl", verl.__version__, verl.__file__)
PY

cat <<EOF

Environment ready at: $ENV
  export PATH=$ENV/bin:\$PATH      # or per-command: $ENV/bin/python ...
Required runtime exports (also baked into the run scripts):
  export PYTHONPATH=<repo>/scripts/vllm_patch:\$PYTHONPATH
  export VLLM_USE_FLASHINFER_SAMPLER=0 FLA_CACHE_RESULTS=0 OMP_NUM_THREADS=16
EOF
