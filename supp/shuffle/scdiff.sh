#!/usr/bin/env bash
# CD4T known-condition | shuffle gene order | scDiff | GPU 3
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}" >&2' ERR

source "$(dirname "$0")/../common/lib.sh"
init_supp shuffle 3
require_data

METHOD_NAME="scDiff"
NAME="${NAME:-v7.5}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
SAMPLE_DIR="${SAMPLE_BASE}/scdiff_${NUM_GENES}"
CSV_PATH="${SAMPLE_DIR}/metrics_${METHOD_NAME}_${CELL_TYPE}_hvg_${NUM_GENES}.csv"
mkdir -p "${SAMPLE_DIR}"

dataset_name="fig1_task1_${CELL_TYPE}"
train_fname="${CELL_TYPE}_train_HVG_${NUM_GENES}.h5ad"
valid_fname="${CELL_TYPE}_valid_HVG_${NUM_GENES}.h5ad"

base_data_settings=()
base_data_settings+=("data.params.train.params.dataset=${dataset_name}")
base_data_settings+=("data.params.train.params.fname=${train_fname}")
base_data_settings+=("data.params.test.params.dataset=${dataset_name}")
base_data_settings+=("data.params.test.params.fname=${valid_fname}")
base_data_settings+=("model.params.generation_kwargs.n_samples=${N_SAMPLES}")

all_outputs=""
for (( i=1; i<=NUM_RUNS; i++ )); do
  echo "=== Run ${i}/${NUM_RUNS} | ${GENE_ORDER} | ${METHOD_NAME} ==="
  run_postfix="perturbation_${NAME}_run${i}"
  model_save_path="${CKPT_BASE}/scdiff/${CELL_TYPE}_hvg_${NUM_GENES}/run${i}"

  output=$(
    python src/scDiff/main.py \
      --custom_data_path "${DATA_DIR}" \
      --base configs/scdiff/eval_perturbation.yaml \
      --name "${NAME}" \
      --logdir "${LOG_DIR}" \
      --postfix "${run_postfix}" \
      --model_save_path "${model_save_path}" \
      ${OFFLINE_SETTINGS} \
      "${base_data_settings[@]}" 2>&1
  ) || true
  echo "$output"
  all_outputs+="$output\n"
done

aggregate_metrics "${METHOD_NAME}" "${CSV_PATH}" "${all_outputs}"
echo "Done: ${GENE_ORDER} ${METHOD_NAME}"
