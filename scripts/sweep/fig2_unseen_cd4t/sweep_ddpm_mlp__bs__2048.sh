#!/bin/bash
# DDPM+MLP batch_size=2048 (default)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm_mlp
export SWEEP_AXIS=bs
export SWEEP_TAG=2048
export SWEEP_BS=2048
source "${SCRIPT_DIR}/_common.sh"
