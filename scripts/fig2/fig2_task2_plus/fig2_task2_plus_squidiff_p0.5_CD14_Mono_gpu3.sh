#!/usr/bin/env bash
# Temporary: p0.5 only, holdout CD14+Mono, GPU 3.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=3
export FIG2_TASK2_PLUS_SQUIDIFF_SLUGS="p0.5"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="CD14+Mono"
exec bash "${DIR}/fig2_task2_plus_squidiff.sh"
