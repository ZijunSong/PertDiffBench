#!/bin/bash
# Fig4 时间条件生成 — scDiff（训练使用 fig4 数据；时间条件采样需扩展 cond 为 treatment_time，完成后可接 eval_fig4）

set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiff}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"

HOMEDIR=$(dirname "$(dirname "$(realpath "$0")")")
cd "$HOMEDIR"

TRAIN_H5="data/fig4/fig4_train.h5ad"
TEST_H5="data/fig4/fig4_test.h5ad"
# scDiff 需要 config + custom_data_path；fig4 数据需单独 config 或覆盖 data.params
dataset_name="fig4"
train_fname="fig4_train.h5ad"
valid_fname="fig4_test.h5ad"
data_dir="data/fig4"

sample_base="samples/fig4/scdiff_3000"
csv_path="${sample_base}/metrics_${METHOD_NAME}_fig4.csv"
log_file="${LOGDIR}/fig4_task1/scdiff_fig4.log"
mkdir -p "${sample_base}" "${LOGDIR}/fig4_task1"

# 检查 fig4 专用 config 是否存在；若无则用默认并覆盖 data path
CONFIG_BASE="configs/scdiff/eval_perturbation.yaml"
BASE_DATA_SETTINGS=(
  "data.params.train.params.dataset=${dataset_name}"
  "data.params.train.params.fname=${train_fname}"
  "data.params.test.params.dataset=${dataset_name}"
  "data.params.test.params.fname=${valid_fname}"
)

{
  echo "== $(date '+%F %T') | fig4 scDiff | runs=${NUM_RUNS} =="
  all_outputs=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo "====================== Run ${i}/${NUM_RUNS} ======================"
    run_postfix="fig4_run${i}"
    output=$(python src/scDiff/main.py \
      --custom_data_path "${data_dir}" \
      --base "${CONFIG_BASE}" \
      --name "fig4" \
      --logdir "${LOGDIR}" \
      --postfix "${run_postfix}" \
      ${OFFLINE_SETTINGS} \
      "${BASE_DATA_SETTINGS[@]}" 2>&1) || true
    echo "$output"
    all_outputs+="$output\n"
  done

  echo ""
  echo "Note: scDiff 时间条件采样（生成 4h/6h）需在 data/model 中增加 treatment_time 条件。"
  echo "完成后可运行: python scripts/fig4/eval_fig4_time_conditioned.py --test-h5ad ${TEST_H5} --generated-h5ad <path> --train-h5ad ${TRAIN_H5}"
  echo "--- Finished fig4 scDiff (train only; time-conditioned sampling TBD) ---"
} 2>&1 | tee -a "${log_file}"

echo "######################################################################"
echo "###   fig4_task1 scDiff complete (train only).                     ###"
echo "######################################################################"
