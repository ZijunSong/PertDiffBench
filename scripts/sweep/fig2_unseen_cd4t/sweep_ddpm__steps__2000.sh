#!/bin/bash
# DDPM diffusion_steps=2000 (higher)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm
export SWEEP_AXIS=steps
export SWEEP_TAG=2000
export SWEEP_STEPS=2000
source "${SCRIPT_DIR}/_common.sh"
