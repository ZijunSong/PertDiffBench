#!/bin/bash
# DDPM+MLP lr=5e-6 (lower)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=lr
export SWEEP_TAG=5e-6
export SWEEP_LR=5e-6
source "${SCRIPT_DIR}/_common.sh"
