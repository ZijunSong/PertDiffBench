#!/usr/bin/env bash
# CD4T known-condition | cluster gene order | Squidiff | GPU 4
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}" >&2' ERR

source "$(dirname "$0")/../common/lib.sh"
init_supp cluster 4
require_data

METHOD_NAME="Squidiff"
SAVE_BASE="${CKPT_BASE}/squidiff/${CELL_TYPE}_hvg_${NUM_GENES}"
SAMPLE_DIR="${SAMPLE_BASE}/squidiff_${NUM_GENES}"
CSV_PATH="${SAMPLE_DIR}/metrics_${METHOD_NAME}_${CELL_TYPE}_hvg_${NUM_GENES}.csv"
mkdir -p "${SAVE_BASE}" "${SAMPLE_DIR}"

echo "=== Training | ${GENE_ORDER} | ${METHOD_NAME} ==="
python src/Squidiff/train_squidiff.py \
  --logger_path "${LOG_DIR}" \
  --data_path "${TRAIN_H5}" \
  --resume_checkpoint "${SAVE_BASE}" \
  --gene_size "${NUM_GENES}" \
  --output_dim "${NUM_GENES}"

all_outputs=""
for (( i=1; i<=NUM_RUNS; i++ )); do
  export RUN_SEED=$(($i-1))
  echo "=== Inference ${i}/${NUM_RUNS} | ${GENE_ORDER} | ${METHOD_NAME} ==="
  sample_dir_run="${SAMPLE_DIR}/run${i}"
  mkdir -p "${sample_dir_run}"
  output=$(
    python src/Squidiff/sample_squidiff.py \
      --model_path "${SAVE_BASE}/model.pt" \
      --gene_size "${NUM_GENES}" \
      --output_dim "${NUM_GENES}" \
      --out_h5ad "${sample_dir_run}/synthetic_ifn_run_${i}.h5ad" \
      --train_data_path "${VALID_H5}" \
      --n_samples "${N_SAMPLES}" \
      --umap_plot "${sample_dir_run}/umap_comparison_${i}.png" \
      --data_path "${VALID_H5}" 2>&1
  ) || true
  echo "$output"
  all_outputs+="$output\n"
done

aggregate_metrics "${METHOD_NAME}" "${CSV_PATH}" "${all_outputs}"
echo "Done: ${GENE_ORDER} ${METHOD_NAME}"
