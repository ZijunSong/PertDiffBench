#!/usr/bin/env bash
# DDPM MOA Diff-MOA: SMILES + dose conditioning (Squidiff-style, default)
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
export PYTHONUNBUFFERED=1
IFS=$'\n\t'
trap 'echo ERROR && exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GENES="${NUM_GENES:-3000}"
NUM_RUNS="${NUM_RUNS:-3}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-32}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-1000}"
SKIP_UMAP="${SKIP_UMAP:-true}"

HOMEDIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$HOMEDIR"
export PYTHONPATH="${HOMEDIR}:${PYTHONPATH:-}"

USE_DRUG_STRUCTURE="${USE_DRUG_STRUCTURE:-true}"

DATA_BASE="${DATA_BASE:-/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA/control_plus_ifn_with_smiles}"
SAMPLES_BASE="${SAMPLES_BASE:-/data/ppnm/data/PertDiffBench/samples}"
CKPT_BASE="${CKPT_BASE:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"

DATA_ROOT="${DATA_ROOT:-${DATA_BASE}/control_plus_ifn/unseen_diff_moa}"
CONFIG_FILE="${CONFIG_FILE:-configs/baselines/scrna_ddpm_scrna.yaml}"
SAMPLES_ROOT="${SAMPLES_ROOT:-${SAMPLES_BASE}/fig2/task1_unseenMOA/diff}"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_BASE}/fig2/task1_unseenMOA/diff}"

mkdir -p "${SAMPLES_ROOT}" "${CKPT_ROOT}"

DRUG_STRUCT_ARGS=()
if [[ "${USE_DRUG_STRUCTURE}" == "true" ]]; then
  DRUG_STRUCT_ARGS=(--use-drug-structure)
fi

mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_train__plus_control.h5ad" 2>/dev/null | sort)
[[ ${#TRAIN_FILES[@]} -gt 0 ]] || { echo "[ERROR] No *_train__plus_control.h5ad under ${DATA_ROOT}" >&2; exit 1; }

echo "Found ${#TRAIN_FILES[@]} MOA datasets | runs=${NUM_RUNS} | genes=${NUM_GENES} | batch=${BATCH_SIZE} | workers=${NUM_WORKERS} | use_drug_structure=${USE_DRUG_STRUCTURE}"
echo

for train_path in "${TRAIN_FILES[@]}"; do
  train_fname="$(basename "${train_path}")"
  moa="${train_fname%_train__plus_control.h5ad}"
  test_path="${DATA_ROOT}/${moa}_test__plus_control.h5ad"
  [[ -f "${test_path}" ]] || { echo "[WARN] Skip ${moa}: missing test" >&2; continue; }

  echo "######################################################################"
  echo "###   DDPM MOA: ${moa} (${NUM_RUNS} runs)"
  echo "######################################################################"

  ckpt_base="${CKPT_ROOT}/ddpm/${moa}_${NUM_GENES}"
  sample_base="${SAMPLES_ROOT}/${moa}/DDPM_${NUM_GENES}"
  csv_dir="${sample_base}/metrics"
  mkdir -p "${ckpt_base}" "${sample_base}" "${csv_dir}"
  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo "--- Run ${i}/${NUM_RUNS} for ${moa} ---"
    run_ckpt="${ckpt_base}/run${i}"
    run_sample="${sample_base}/run${i}"
    mkdir -p "${run_ckpt}" "${run_sample}"

    ckpt_pt="${run_ckpt}/scrna_ddpm_epoch1000.pt"
    label_enc="${run_ckpt}/label_encoder.npz"

    python scripts/baseline_exp/train_scrna_ddpm_scrna_moa.py \
      --config "${CONFIG_FILE}" \
      --data-path "${train_path}" \
      --save-weight-dir "${run_ckpt}" \
      --gene-nums "${NUM_GENES}" \
      --drug-key perturbation \
      --dose-key dose_value \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --ckpt-save-interval "${CKPT_SAVE_INTERVAL}" \
      "${DRUG_STRUCT_ARGS[@]}"

    [[ -f "${ckpt_pt}" ]] || { echo "[ERROR] Checkpoint not found: ${ckpt_pt}" >&2; exit 1; }

    UMAP_ARGS=()
    if [[ "${SKIP_UMAP}" != "true" ]]; then
      UMAP_ARGS=(--umap_plot "${run_sample}/umap_comparison_${i}.png")
    fi

    run_out="$(
      python scripts/baseline_exp/eval_scrna_ddpm_scrna_moa.py \
        --config "${CONFIG_FILE}" \
        --ckpt "${ckpt_pt}" \
        --label-encoder-path "${label_enc}" \
        --data-path "${test_path}" \
        --train-data-path "${train_path}" \
        --n_samples "${NUM_SAMPLES}" \
        --out_h5ad "${run_sample}/synthetic_${moa}_${i}.h5ad" \
        --gene-nums "${NUM_GENES}" \
        --drug-key perturbation \
        --dose-key dose_value \
        "${UMAP_ARGS[@]}" \
        "${DRUG_STRUCT_ARGS[@]}" 2>&1
    )" || { echo "[ERROR] Eval failed for run ${i}" >&2; exit 1; }
    echo "${run_out}"
    ALL_OUTPUTS+="${run_out}"$'\n'
  done

  printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="${moa}_test" -v num_runs="${NUM_RUNS}" -v method="DDPM(${NUM_GENES})" -v csv_path="${csv_dir}/metrics_${moa}.csv" '
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }
    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
    /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
    /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
    /^E-Distance:/  { edist[c_edist++] = to_num($NF) }
    /Maximum Mean Discrepancy \(MMD\):/  { mmd[c_mmd++] = to_num($NF) }
    /R-squared \(R2\):/  { r2[c_r2++] = to_num($NF) }
    /Pearson \(all genes\):/  { p_all[c_p_all++] = to_num($NF) }
    /Pearson Delta \(all genes\):/  { pd_all[c_pd_all++] = to_num($NF) }
    /Pearson Delta \(top 20 DE genes\):/  { pd20[c_pd20++] = to_num($NF) }
    /Pearson Delta \(top 50 DE genes\):/  { pd50[c_pd50++] = to_num($NF) }
    /Pearson Delta \(top 100 DE genes\):/  { pd100[c_pd100++] = to_num($NF) }
    function mean(a,n,s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n?s/n:0 }
    function std(a,n,mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)^2; return sqrt(s/(n-1)) }
    END {
      row=ds","method
      row=row","sprintf("%.6f±%.6f", mean(pds,c_pds), std(pds,c_pds))
      row=row","sprintf("%.6f±%.6f", mean(mae,c_mae), std(mae,c_mae))
      row=row","sprintf("%.6f±%.6f", mean(des,c_des), std(des,c_des))
      row=row","sprintf("%.6f±%.6f", mean(edist,c_edist), std(edist,c_edist))
      row=row","sprintf("%.6f±%.6f", mean(mmd,c_mmd), std(mmd,c_mmd))
      row=row","sprintf("%.6f±%.6f", mean(r2,c_r2), std(r2,c_r2))
      row=row","sprintf("%.6f±%.6f", mean(p_all,c_p_all), std(p_all,c_p_all))
      row=row","sprintf("%.6f±%.6f", mean(pd_all,c_pd_all), std(pd_all,c_pd_all))
      row=row","sprintf("%.6f±%.6f", mean(pd20,c_pd20), std(pd20,c_pd20))
      row=row","sprintf("%.6f±%.6f", mean(pd50,c_pd50), std(pd50,c_pd50))
      row=row","sprintf("%.6f±%.6f", mean(pd100,c_pd100), std(pd100,c_pd100))
      print row >> csv_path
      close(csv_path)
      printf("CSV: %s\n", csv_path)
    }
  '

  echo "--- Finished MOA: ${moa} ---"
  echo
done

echo "######################################################################"
echo "### All DDPM MOAs completed! CSVs under ${SAMPLES_ROOT}/*/DDPM_*/metrics/"
echo "######################################################################"

# ---- Step: Aggregate all MOA CSV results ----
echo
echo "######################################################################"
echo "###   Aggregating all MOA results into a single CSV file..."
echo "######################################################################"

AGGREGATED_CSV="${SAMPLES_ROOT}/aggregated_metrics_DDPM_${NUM_GENES}.csv"
python3 "${HOMEDIR}/utils/aggregate_metrics.py" \
  --samples-root "${SAMPLES_ROOT}" \
  --output-csv "${AGGREGATED_CSV}" \
  --pattern "DDPM_${NUM_GENES}"

if [[ -f "${AGGREGATED_CSV}" ]]; then
  echo
  echo "######################################################################"
  echo "###   Aggregation completed!"
  echo "###   Aggregated CSV: ${AGGREGATED_CSV}"
  echo "###   Absolute path: $(cd "$(dirname "${AGGREGATED_CSV}")" && pwd)/$(basename "${AGGREGATED_CSV}")"
  echo "######################################################################"
else
  echo "[WARN] Failed to create aggregated CSV file"
fi
