#!/usr/bin/env bash

set -euo pipefail

RAW_DIR="/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseen_pert"
OUT_DIR="/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenPert"

mkdir -p "${OUT_DIR}"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/dfcontrol_exp.csv" \
    --meta "${RAW_DIR}/dfcontrol_meta.csv" \
    --output "${OUT_DIR}/dfcontrol.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed123_test_exp.csv" \
    --meta "${RAW_DIR}/seed123_test_meta.csv" \
    --output "${OUT_DIR}/seed123_test.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed123_train_exp.csv" \
    --meta "${RAW_DIR}/seed123_train_meta.csv" \
    --output "${OUT_DIR}/seed123_train.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed345_test_exp.csv" \
    --meta "${RAW_DIR}/seed345_test_meta.csv" \
    --output "${OUT_DIR}/seed345_test.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed345_train_exp.csv" \
    --meta "${RAW_DIR}/seed345_train_meta.csv" \
    --output "${OUT_DIR}/seed345_train.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed567_test_exp.csv" \
    --meta "${RAW_DIR}/seed567_test_meta.csv" \
    --output "${OUT_DIR}/seed567_test.h5ad"

python scripts/tools/fig2_task1_merge.py \
    --exp "${RAW_DIR}/seed567_train_exp.csv" \
    --meta "${RAW_DIR}/seed567_train_meta.csv" \
    --output "${OUT_DIR}/seed567_train.h5ad"

python scripts/tools/fig2_task1_split.py \
    --control "${OUT_DIR}/dfcontrol.h5ad" \
    --train "${OUT_DIR}/seed123_train.h5ad" \
    --test "${OUT_DIR}/seed123_test.h5ad" \
    --output_train "${OUT_DIR}/seed123_control_train.h5ad" \
    --output_test "${OUT_DIR}/seed123_control_test.h5ad"

python scripts/tools/fig2_task1_split.py \
    --control "${OUT_DIR}/dfcontrol.h5ad" \
    --train "${OUT_DIR}/seed345_train.h5ad" \
    --test "${OUT_DIR}/seed345_test.h5ad" \
    --output_train "${OUT_DIR}/seed345_control_train.h5ad" \
    --output_test "${OUT_DIR}/seed345_control_test.h5ad"

python scripts/tools/fig2_task1_split.py \
    --control "${OUT_DIR}/dfcontrol.h5ad" \
    --train "${OUT_DIR}/seed567_train.h5ad" \
    --test "${OUT_DIR}/seed567_test.h5ad" \
    --output_train "${OUT_DIR}/seed567_control_train.h5ad" \
    --output_test "${OUT_DIR}/seed567_control_test.h5ad"
