#!/bin/bash
# DDPM diffusion_steps=1000 (default)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm
export SWEEP_AXIS=steps
export SWEEP_TAG=1000
export SWEEP_STEPS=1000
source "${SCRIPT_DIR}/_common.sh"
