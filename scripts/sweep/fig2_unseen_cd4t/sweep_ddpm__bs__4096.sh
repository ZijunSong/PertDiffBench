#!/bin/bash
# DDPM batch_size=4096 (higher)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm
export SWEEP_AXIS=bs
export SWEEP_TAG=4096
export SWEEP_BS=4096
source "${SCRIPT_DIR}/_common.sh"
