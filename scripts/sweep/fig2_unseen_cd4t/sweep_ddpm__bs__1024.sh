#!/bin/bash
# DDPM batch_size=1024 (lower)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_METHOD=ddpm
export SWEEP_AXIS=bs
export SWEEP_TAG=1024
export SWEEP_BS=1024
source "${SCRIPT_DIR}/_common.sh"
