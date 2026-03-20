#!/usr/bin/env bash
# Run from PertDiffBench repo root. CSV 来自 data_ori；合并后的 H5AD 写入 data/fig1_task4（与 fig1_task1–3 一致）
DATA_ORI="/data/ppnm/data/PertDiffBench/data_ori/fig1/task4"
DATA_OUT="/data/ppnm/data/PertDiffBench/data/fig1_task4"
mkdir -p "$DATA_OUT"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_ACTA2_control_meta.csv" \
    "${DATA_ORI}/task4_ACTA2_hvg3000_lognorm_control_exp.csv" \
    "${DATA_OUT}/task4_ACTA2_control.h5ad"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_ACTA2_coculture_meta.csv" \
    "${DATA_ORI}/task4_ACTA2_hvg3000_lognorm_coculture_exp.csv" \
    "${DATA_OUT}/task4_ACTA2_coculture.h5ad"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_ACTA2_ifn_meta.csv" \
    "${DATA_ORI}/task4_ACTA2_hvg3000_lognorm_ifn_exp.csv" \
    "${DATA_OUT}/task4_ACTA2_ifn.h5ad"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_B2M_control_meta.csv" \
    "${DATA_ORI}/task4_B2M_hvg3000_lognorm_control_exp.csv" \
    "${DATA_OUT}/task4_B2M_control.h5ad"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_B2M_coculture_meta.csv" \
    "${DATA_ORI}/task4_B2M_hvg3000_lognorm_coculture_exp.csv" \
    "${DATA_OUT}/task4_B2M_coculture.h5ad"

python scripts/tools/fig1_task4_merge.py \
    "${DATA_ORI}/task4_B2M_ifn_meta.csv" \
    "${DATA_ORI}/task4_B2M_hvg3000_lognorm_ifn_exp.csv" \
    "${DATA_OUT}/task4_B2M_ifn.h5ad"
