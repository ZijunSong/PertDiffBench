#!/usr/bin/env bash
# Shared helpers for supp gene-order CD4T experiments.

set -euo pipefail

init_supp() {
  local order="$1"
  local gpu="$2"
  export GENE_ORDER="${order}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export PYTHONUNBUFFERED=1

  SUPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  HOMEDIR="$(cd "${SUPP_DIR}/.." && pwd)"
  cd "${HOMEDIR}"
  export PYTHONPATH="${HOMEDIR}:${PYTHONPATH:-}"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate pertdiffbench 2>/dev/null || true
  fi

  ROOT_DIR="${ROOT_DIR:-/data/ppnm/data/PertDiffBench/}"
  CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"

  CELL_TYPE="CD4T"
  NUM_GENES="1000"
  NUM_RUNS="${NUM_RUNS:-3}"
  N_SAMPLES=278

  DATA_DIR="${ROOT_DIR}data/gene_order_exp/${GENE_ORDER}/"
  TRAIN_H5="${DATA_DIR}${CELL_TYPE}_train_HVG_${NUM_GENES}.h5ad"
  VALID_H5="${DATA_DIR}${CELL_TYPE}_valid_HVG_${NUM_GENES}.h5ad"

  CKPT_BASE="${CKPT_ROOT}/supp/${GENE_ORDER}"
  SAMPLE_BASE="${ROOT_DIR}samples/supp/${GENE_ORDER}/${CELL_TYPE}"
  LOG_DIR="${SUPP_DIR}/logs/${GENE_ORDER}"
  AWK_FILE="${SUPP_DIR}/common/aggregate_metrics.awk"

  mkdir -p "${LOG_DIR}" "${SAMPLE_BASE}" "${CKPT_BASE}"
}

require_data() {
  if [[ ! -f "${TRAIN_H5}" || ! -f "${VALID_H5}" ]]; then
    echo "Missing reordered h5ad. Run first:"
    echo "  python supp/preprocess_reorder_genes_cd4t.py --mode both"
    exit 1
  fi
}

aggregate_metrics() {
  local method="$1"
  local csv_path="$2"
  local all_outputs="$3"
  echo -e "${all_outputs}" | awk \
    -v dataset="${CELL_TYPE}" \
    -v num_runs="${NUM_RUNS}" \
    -v method="${method}" \
    -v csv_path="${csv_path}" \
    -f "${AWK_FILE}"
}
