#!/bin/bash
# DDPM+MLP diffusion_steps=500 (lower)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=steps
export SWEEP_TAG=500
export SWEEP_STEPS=500
source "${SCRIPT_DIR}/_common.sh"
