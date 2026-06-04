#!/usr/bin/env bash
# CD4T known-condition | cluster gene order | scDiffusion | GPU 5
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}" >&2' ERR

source "$(dirname "$0")/../common/lib.sh"
init_supp cluster 5
require_data

METHOD_NAME="scDiffusion"
ANNOTATION_MODEL_DIR="${ANNOTATION_MODEL_DIR:-/data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1}"
VAE_BASE="${CKPT_BASE}/scdiffusion/vae_checkpoint/${CELL_TYPE}_${NUM_GENES}"
DIFF_BASE="${CKPT_BASE}/scdiffusion/diffusion_checkpoint/${CELL_TYPE}_${NUM_GENES}"
CLS_BASE="${CKPT_BASE}/scdiffusion/classifier_checkpoint/${CELL_TYPE}_${NUM_GENES}"
SAMPLE_DIR="${SAMPLE_BASE}/scDiffusion_${NUM_GENES}"
CSV_PATH="${SAMPLE_DIR}/metrics_${METHOD_NAME}_${CELL_TYPE}_hvg_${NUM_GENES}.csv"
mkdir -p "${VAE_BASE}" "${DIFF_BASE}" "${CLS_BASE}" "${SAMPLE_DIR}"

all_outputs=""
for (( i=1; i<=NUM_RUNS; i++ )); do
  echo "=== Run ${i}/${NUM_RUNS} | ${GENE_ORDER} | ${METHOD_NAME} ==="
  vae_dir="${VAE_BASE}/run${i}"
  diff_dir="${DIFF_BASE}/run${i}"
  cls_dir="${CLS_BASE}/run${i}"
  sample_dir_run="${SAMPLE_DIR}/run${i}"
  mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${sample_dir_run}"

  vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
  diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
  cls_ckpt="${cls_dir}/model009999.pt"

  pushd src/scDiffusion/VAE >/dev/null
  python VAE_train.py \
    --data_dir "${TRAIN_H5}" \
    --num_genes "${NUM_GENES}" \
    --state_dict "${ANNOTATION_MODEL_DIR}" \
    --save_dir "${vae_dir}"
  popd >/dev/null

  pushd src/scDiffusion >/dev/null
  python cell_train.py \
    --data_dir "${TRAIN_H5}" \
    --vae_path "${vae_ckpt}" \
    --save_dir "${diff_dir}"

  python classifier_train.py \
    --data_dir "${TRAIN_H5}" \
    --vae_path "${vae_ckpt}" \
    --model_path "${cls_dir}"

  output=$(
    python classifier_sample.py \
      --num_samples "${N_SAMPLES}" \
      --train-data-path "${TRAIN_H5}" \
      --model_path "${diff_ckpt}" \
      --classifier_path "${cls_ckpt}" \
      --ae_dir "${vae_ckpt}" \
      --num_gene "${NUM_GENES}" \
      --sample_dir "${sample_dir_run}" \
      --out_h5ad "${sample_dir_run}/synthetic_ifn_${i}.h5ad" \
      --umap_plot "${sample_dir_run}/umap_comparison_${i}.png" \
      --init_cell_path "${VALID_H5}" 2>&1
  ) || true
  popd >/dev/null

  echo "$output"
  all_outputs+="$output\n"
done

aggregate_metrics "${METHOD_NAME}" "${CSV_PATH}" "${all_outputs}"
echo "Done: ${GENE_ORDER} ${METHOD_NAME}"
