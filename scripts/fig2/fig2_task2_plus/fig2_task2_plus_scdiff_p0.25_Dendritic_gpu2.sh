#!/usr/bin/env bash
# Temporary: p0.25 only, holdout Dendritic, GPU 2.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=2
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.25"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="Dendritic"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
