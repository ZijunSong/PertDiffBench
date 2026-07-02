#!/usr/bin/env bash
# Temporary: NK @ p0, GPU 4.
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=4
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="NK"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
