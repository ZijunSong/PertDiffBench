#!/usr/bin/env bash
# Temporary: FCGR3A+Mono @ p0.25, GPU 3.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=3
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.25"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="FCGR3A+Mono"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
