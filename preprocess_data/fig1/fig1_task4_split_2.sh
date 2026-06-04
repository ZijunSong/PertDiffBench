#!/usr/bin/env bash
# Run from PertDiffBench repo root. Input/output under fig1_task4 H5AD directory
DATA_OUT="/data/ppnm/data/PertDiffBench/data/fig1_task4"
mkdir -p "$DATA_OUT"

python scripts/tools/fig1_task4_split_2.py \
    "${DATA_OUT}/task4_ACTA2_control.h5ad" \
    "${DATA_OUT}/task4_ACTA2_coculture.h5ad" \
    "${DATA_OUT}/task4_ACTA2_ifn.h5ad" \
    "${DATA_OUT}/task4_ACTA2_control_to_coculture.h5ad" \
    "${DATA_OUT}/task4_ACTA2_control_to_ifn.h5ad"

python scripts/tools/fig1_task4_split_2.py \
    "${DATA_OUT}/task4_B2M_control.h5ad" \
    "${DATA_OUT}/task4_B2M_coculture.h5ad" \
    "${DATA_OUT}/task4_B2M_ifn.h5ad" \
    "${DATA_OUT}/task4_B2M_control_to_coculture.h5ad" \
    "${DATA_OUT}/task4_B2M_control_to_ifn.h5ad"
