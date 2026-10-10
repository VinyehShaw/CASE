#!/bin/bash
# Stage 0 - Environment.
#
# Two supported, equivalent routes (see README >> Environment):
#   (a) docker:  docker pull <registry>/case:<tag>
#   (b) bare metal: install the released conda-env tar (Hugging Face):
#           ENV_TAR=/path/to/case_env_0904.tar bash run/00_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"
TAR="${ENV_TAR:-$CODE_ROOT/env_release/case_env_0904.tar}"
if [ ! -f "$TAR" ]; then
    cat <<EOF
env tar not found: $TAR
Download the released environment tar (see README >> Environment), then run:
    ENV_TAR=/path/to/case_env_0904.tar bash run/00_env.sh
Or use the docker image instead:
    docker pull <registry>/case:<tag>
EOF
    exit 1
fi
bash "$CODE_ROOT/env_release/install_env_tar.sh" "$TAR"
