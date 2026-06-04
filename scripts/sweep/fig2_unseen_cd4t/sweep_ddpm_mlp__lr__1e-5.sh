#!/bin/bash
# DDPM+MLP lr=1e-5 (default)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=lr
export SWEEP_TAG=1e-5
export SWEEP_LR=1e-5
source "${SCRIPT_DIR}/_common.sh"
