#!/bin/bash
# Fig4 time-conditioned generation - scDiff (train using fig4 data; when rows need cond as treatment_time, doneaftercan eval_fig4)

set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

source "scripts/lib/max_n_samples.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiff}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"

HOMEDIR=$(dirname "$(dirname "$(realpath "$0")")")
cd "$HOMEDIR"

DATA_FIG4="/data/ppnm/data/PertDiffBench/data/fig4_task1"
TRAIN_H5="${DATA_FIG4}/fig4_train.h5ad"

TEST_H5="${TRAIN_H5/fig4_train/fig4_test}"
N_SAMPLES="$(max_n_samples_timepoint "${TEST_H5:-data/fig4_task1/fig4_test.h5ad}")"
TEST_H5="${DATA_FIG4}/fig4_test.h5ad"
# scDiff must config + custom_data_path; fig4 dataneeds separate config or data.params
dataset_name="fig4"
train_fname="fig4_train.h5ad"
valid_fname="fig4_test.h5ad"
data_dir="${DATA_FIG4}"

sample_base="samples/fig4/scdiff_3000"
csv_path="${sample_base}/metrics_${METHOD_NAME}_fig4.csv"
log_file="${LOGDIR}/fig4_task1/scdiff_fig4.log"
mkdir -p "${sample_base}" "${LOGDIR}/fig4_task1"

# check fig4 using config whetherexist; usingdefaultand data path
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
    export RUN_SEED=$(($i-1))
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
  echo "Note: scDiff when rows ( 4h/6h)needin data/model treatment_time rows ."
  echo "doneaftercan : python scripts/fig4/eval_fig4_time_conditioned.py --test-h5ad ${TEST_H5} --generated-h5ad <path> --train-h5ad ${TRAIN_H5}"
  echo "--- Finished fig4 scDiff (train only; time-conditioned sampling TBD) ---"
} 2>&1 | tee -a "${log_file}"

echo "######################################################################"
echo "###   fig4_task1 scDiff complete (train only).                     ###"
echo "######################################################################"
