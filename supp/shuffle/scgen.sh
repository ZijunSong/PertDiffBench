#!/usr/bin/env bash
# CD4T known-condition | shuffle gene order | scGen | GPU 2
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}" >&2' ERR

source "$(dirname "$0")/../common/lib.sh"
init_supp shuffle 2
require_data

METHOD_NAME="scGen"
SAVE_BASE="${CKPT_BASE}/scgen/${CELL_TYPE}_hvg_${NUM_GENES}"
SAMPLE_DIR="${SAMPLE_BASE}/scgen_${NUM_GENES}"
CSV_PATH="${SAMPLE_DIR}/metrics_${METHOD_NAME}_${CELL_TYPE}_hvg_${NUM_GENES}.csv"
mkdir -p "${SAVE_BASE}" "${SAMPLE_DIR}"

all_outputs=""
for (( i=1; i<=NUM_RUNS; i++ )); do
  export RUN_SEED=$(($i-1))
  echo "=== Run ${i}/${NUM_RUNS} | ${GENE_ORDER} | ${METHOD_NAME} ==="
  save_dir_run="${SAVE_BASE}/run${i}"
  sample_dir_run="${SAMPLE_DIR}/run${i}"
  mkdir -p "${save_dir_run}" "${sample_dir_run}"

  output=$(
    python scripts/scGen_eval.py \
      --train_data_path "${TRAIN_H5}" \
      --test_data_path "${VALID_H5}" \
      --model_save_path "${save_dir_run}" \
      --out_h5ad "${sample_dir_run}/${CELL_TYPE}_${NUM_GENES}_pred_${i}.h5ad" \
      --umap_plot "${sample_dir_run}/${CELL_TYPE}_umap_comparison_${i}.png" \
      --n_samples "${N_SAMPLES}" \
      --celltype_to_predict "${CELL_TYPE}" 2>&1
  ) || true
  echo "$output"
  all_outputs+="$output\n"
done

aggregate_metrics "${METHOD_NAME}" "${CSV_PATH}" "${all_outputs}"
echo "Done: ${GENE_ORDER} ${METHOD_NAME}"
