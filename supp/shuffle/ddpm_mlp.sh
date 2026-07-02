#!/usr/bin/env bash
# CD4T known-condition | shuffle gene order | DDPM+MLP | GPU 0
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}" >&2' ERR

source "$(dirname "$0")/../common/lib.sh"
init_supp shuffle 0
require_data

METHOD_NAME="DDPM+MLP"
CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"
CKPT_NAME="model_epoch_1000.pth"
SAVE_BASE="${CKPT_BASE}/ddpm_mlp/${CELL_TYPE}_hvg_${NUM_GENES}"
SAMPLE_DIR="${SAMPLE_BASE}/mlp_ddpm_mlp_${NUM_GENES}"
CSV_PATH="${SAMPLE_DIR}/metrics_mlp_ddpm_mlp_${CELL_TYPE}_hvg_${NUM_GENES}.csv"
mkdir -p "${SAVE_BASE}" "${SAMPLE_DIR}"

all_outputs=""
for (( run_idx=1; run_idx<=NUM_RUNS; run_idx++ )); do
  export RUN_SEED=$(($run_idx-1))
  echo "=== Run ${run_idx}/${NUM_RUNS} | ${GENE_ORDER} | ${METHOD_NAME} ==="
  save_dir_run="${SAVE_BASE}/run${run_idx}"
  sample_dir_run="${SAMPLE_DIR}/run${run_idx}"
  mkdir -p "${save_dir_run}" "${sample_dir_run}"
  ckpt_path="${save_dir_run}/${CKPT_NAME}"

  python scripts/baseline_exp/train_mlp_ddpm_mlp.py \
    --config "${CONFIG_FILE}" \
    --data-path "${TRAIN_H5}" \
    --save-weight-dir "${save_dir_run}" \
    --gene-nums "${NUM_GENES}"

  output=$(
    python scripts/baseline_exp/eval_mlp_ddpm_mlp.py \
      --config "${CONFIG_FILE}" \
      --train-data-path "${TRAIN_H5}" \
      --data-path "${VALID_H5}" \
      --ckpt "${ckpt_path}" \
      --out_h5ad "${sample_dir_run}/synthetic_ifn_run${run_idx}.h5ad" \
      --umap_plot "${sample_dir_run}/umap_comparison_${run_idx}.png" \
      --n_samples "${N_SAMPLES}" \
      --gene-nums "${NUM_GENES}" 2>&1
  ) || true
  echo "$output"
  all_outputs+="$output\n"
done

aggregate_metrics "${METHOD_NAME}" "${CSV_PATH}" "${all_outputs}"
echo "Done: ${GENE_ORDER} ${METHOD_NAME}"
