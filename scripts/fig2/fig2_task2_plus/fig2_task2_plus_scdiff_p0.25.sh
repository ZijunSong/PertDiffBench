#!/usr/bin/env bash
# Fig2 task2+ scDiff: control fraction p0.25 only. Default GPU 1; override with CUDA_VISIBLE_DEVICES.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.25"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
