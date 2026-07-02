#!/usr/bin/env bash
# Temporary: p0.25 only, holdout CD8T, GPU 0.
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.25"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="CD8T"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
