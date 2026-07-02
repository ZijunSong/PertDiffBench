#!/usr/bin/env bash
# Temporary: p0.5 only, holdout Dendritic, GPU 4.
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=4
export FIG2_TASK2_PLUS_SQUIDIFF_SLUGS="p0.5"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="Dendritic"
exec bash "${DIR}/fig2_task2_plus_squidiff.sh"
