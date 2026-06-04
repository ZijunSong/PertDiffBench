#!/bin/bash
# DDPM+MLP beta: 1e-4 -> 0.04 (higher)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=beta
export SWEEP_TAG=1e-4_0.04
export SWEEP_BETA1=1e-4
export SWEEP_BETAT=0.04
source "${SCRIPT_DIR}/_common.sh"
