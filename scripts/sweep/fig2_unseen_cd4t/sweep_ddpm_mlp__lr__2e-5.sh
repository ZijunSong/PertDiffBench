#!/bin/bash
# DDPM+MLP lr=2e-5 (higher)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=lr
export SWEEP_TAG=2e-5
export SWEEP_LR=2e-5
source "${SCRIPT_DIR}/_common.sh"
