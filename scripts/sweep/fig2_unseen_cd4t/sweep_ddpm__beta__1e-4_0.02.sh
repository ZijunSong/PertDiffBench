#!/bin/bash
# DDPM beta: 1e-4 -> 0.02 (default)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm
export SWEEP_AXIS=beta
export SWEEP_TAG=1e-4_0.02
export SWEEP_BETA1=1e-4
export SWEEP_BETAT=0.02
source "${SCRIPT_DIR}/_common.sh"
