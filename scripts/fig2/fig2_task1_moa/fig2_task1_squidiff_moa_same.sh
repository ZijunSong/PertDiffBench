#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
IFS=$'\n\t'
trap 'echo "[ERROR] command failed" >&2; exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

# =================== Config ===================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_RUNS="${NUM_RUNS:-3}"
GENE_SIZE="${GENE_SIZE:-3000}"
OUTPUT_DIM="${OUTPUT_DIM:-3000}"
N_SAMPLES="${N_SAMPLES:-100}"
METHOD_NAME="${METHOD_NAME:-Squidiff}"
BATCH_SIZE="${BATCH_SIZE:-3072}"
# 断点续跑：只运行从该 MOA 开始的实验（含）；留空则全部运行。例: START_FROM_MOA=Antioxidant
START_FROM_MOA="${START_FROM_MOA:-}"

DATA_BASE="${DATA_BASE:-/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA/control_plus_ifn_with_smiles}"
SAMPLES_BASE="${SAMPLES_BASE:-/data/ppnm/data/PertDiffBench/samples}"
CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"

DATA_ROOT="${DATA_ROOT:-${DATA_BASE}/control_plus_ifn/unseen_same_moa}"
# 控制样本由 merge_control_with_each_ifn.py 写到 DATA_OUT/control_merged.h5ad
# DATA_OUT = /data/ppnm/data/PertDiffBench/data/fig2/task1_unseenMOA
CONTROL_DATA_PATH="${CONTROL_DATA_PATH:-/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA/control_merged.h5ad}"
[[ -f "${CONTROL_DATA_PATH}" ]] || { echo "[ERROR] Control file not found: ${CONTROL_DATA_PATH}" >&2; exit 1; }

LOGROOT="${LOGROOT:-logs/squidiff}"
OUT_BASE="${OUT_BASE:-${SAMPLES_BASE}/fig2/task1_unseenMOA/same/squidiff}"
CKPT_BASE="${CKPT_ROOT}/fig2/task1_unseenMOA/same/squidiff"
CSV_BASE="${CSV_BASE:-${OUT_BASE}/metrics}"

mkdir -p "${OUT_BASE}" "${CKPT_BASE}" "${CSV_BASE}"

# =================== Discover datasets ===================
mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_train__plus_control.h5ad" | sort)
[[ ${#TRAIN_FILES[@]} -gt 0 ]] || { echo "[ERROR] No *_train__plus_control.h5ad found under: ${DATA_ROOT}" >&2; exit 1; }

echo "Found ${#TRAIN_FILES[@]} MOA datasets under ${DATA_ROOT}"
echo "Using unified control data: ${CONTROL_DATA_PATH}"
echo "Config: runs=${NUM_RUNS} | genes=${GENE_SIZE} | output_dim=${OUTPUT_DIM} | n_samples=${N_SAMPLES}"
echo

# ========================= Main Loop =========================
_resume_reached=false
for train_path in "${TRAIN_FILES[@]}"; do
  train_file="$(basename "${train_path}")"
  moa="${train_file%_train__plus_control.h5ad}"
  test_path="${DATA_ROOT}/${moa}_test__plus_control.h5ad"

  if [[ -n "${START_FROM_MOA:-}" ]]; then
    if [[ "${_resume_reached}" == "false" ]]; then
      if [[ "${moa}" == "${START_FROM_MOA}" ]]; then
        _resume_reached=true
      else
        echo "[SKIP] Skipping MOA ${moa} (resuming from ${START_FROM_MOA})"
        continue
      fi
    fi
  fi

  [[ -f "${test_path}" ]] || { echo "[ERROR] Missing test file for MOA=${moa}: ${test_path}" >&2; exit 1; }

  echo "######################################################################"
  echo "###   Squidiff for MOA: ${moa} (${NUM_RUNS} runs)"
  echo "######################################################################"

  OUT_ROOT="${OUT_BASE}/${moa}"
  CKPT_ROOT="${CKPT_BASE}/${moa}"
  METRICS_CSV="${CSV_BASE}/metrics_${moa}.csv"

  mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"

  # Collect run outputs in a file to avoid argument list length limits
  ALL_OUTPUTS_FILE="${OUT_ROOT}/all_sample_outputs.txt"
  : > "${ALL_OUTPUTS_FILE}"

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for ${moa}"
    echo "======================"

    RUN_CKPT_DIR="${CKPT_ROOT}/run${i}"
    RUN_OUT_DIR="${OUT_ROOT}/run${i}"
    mkdir -p "${RUN_CKPT_DIR}" "${RUN_OUT_DIR}"

    # ---- Step 1: Train (SMILES + shared control) ----
    echo -e "\n--- Training model for ${moa} (run ${i}) ---"
    python src/Squidiff/train_squidiff.py \
      --logger_path "${LOGROOT}/fig2_task1_unseenMOA_${moa}_run${i}" \
      --data_path "${train_path}" \
      --resume_checkpoint "${RUN_CKPT_DIR}" \
      --batch_size "${BATCH_SIZE}" \
      --gene_size "${GENE_SIZE}" \
      --output_dim "${OUTPUT_DIM}" \
      --use_drug_structure True

    echo "--- Training for ${moa} (run ${i}) complete. ---"

    # ---- Step 2: Evaluate ----
    echo -e "\n--- Evaluating (sampling) for ${moa} (run ${i}) ---"

    PRED_H5AD="${RUN_OUT_DIR}/synthetic_${moa}_run_${i}.h5ad"
    UMAP_PNG="${RUN_OUT_DIR}/umap_comparison_${i}.png"
    MODEL_PT="${RUN_CKPT_DIR}/model.pt"

    # Sample: write to temp log then parse to avoid set -e on python non-zero exit
    SAMPLE_LOG="${RUN_OUT_DIR}/sample_log.txt"
    python src/Squidiff/sample_squidiff.py \
      --model_path "${MODEL_PT}" \
      --gene_size "${GENE_SIZE}" \
      --output_dim "${OUTPUT_DIM}" \
      --out_h5ad "${PRED_H5AD}" \
      --n_samples "${N_SAMPLES}" \
      --umap_plot "${UMAP_PNG}" \
      --train_data_path "${train_path}" \
      --data_path "${test_path}" \
      --control_data_path "${CONTROL_DATA_PATH}" \
      --use_drug_structure 2>&1 | tee "${SAMPLE_LOG}" || true
    cat "${SAMPLE_LOG}" 2>/dev/null >> "${ALL_OUTPUTS_FILE}"
    echo "" >> "${ALL_OUTPUTS_FILE}"
  done

  # ---- Step 3: Aggregate metrics and write CSV ----
  echo
  awk -v dataset="${moa}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${METRICS_CSV}" '
    BEGIN { c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=c_p_all=c_pd_all=c_pd20=c_pd50=c_pd100=0 }
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }
    /Perturbation Discrimination Score \(PDS\):/  { pds[c_pds++] = to_num($NF); next }
    /Mean Absolute Error \(MAE\):/                 { mae[c_mae++] = to_num($NF); next }
    /Differential Expression Score \(DES\):/       { des[c_des++] = to_num($NF); next }
    /^E-Distance:/                                { edist[c_edist++] = to_num($NF); next }
    /Maximum Mean Discrepancy \(MMD\):/            { mmd[c_mmd++] = to_num($NF); next }
    /R-squared \(R2\):/                           { r2[c_r2++] = to_num($NF); next }
    /Pearson \(all genes\):/                      { p_all[c_p_all++] = to_num($NF); next }
    /Pearson Delta \(all genes\):/                { pd_all[c_pd_all++] = to_num($NF); next }
    /Pearson Delta \(top 20 DE genes\):/          { pd20[c_pd20++] = to_num($NF); next }
    /Pearson Delta \(top 50 DE genes\):/          { pd50[c_pd50++] = to_num($NF); next }
    /Pearson Delta \(top 100 DE genes\):/         { pd100[c_pd100++] = to_num($NF); next }
    function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n : 0 }
    function std(a,n,  mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }
    function mean_std(idx,  n,mu,sd){
      if(idx==1){ n=c_pds;    mu=mean(pds,n);    sd=std(pds,n) }
      else if(idx==2){ n=c_mae;    mu=mean(mae,n);    sd=std(mae,n) }
      else if(idx==3){ n=c_des;    mu=mean(des,n);    sd=std(des,n) }
      else if(idx==4){ n=c_edist;  mu=mean(edist,n);  sd=std(edist,n) }
      else if(idx==5){ n=c_mmd;    mu=mean(mmd,n);    sd=std(mmd,n) }
      else if(idx==6){ n=c_r2;     mu=mean(r2,n);     sd=std(r2,n) }
      else if(idx==7){ n=c_p_all;  mu=mean(p_all,n);  sd=std(p_all,n) }
      else if(idx==8){ n=c_pd_all; mu=mean(pd_all,n); sd=std(pd_all,n) }
      else if(idx==9){ n=c_pd20;   mu=mean(pd20,n);   sd=std(pd20,n) }
      else if(idx==10){n=c_pd50;   mu=mean(pd50,n);   sd=std(pd50,n) }
      else if(idx==11){n=c_pd100;  mu=mean(pd100,n);  sd=std(pd100,n) }
      return sprintf("%.6f±%.6f", mu, sd)
    }
    function val(idx, r, v){
      if(idx==1) v=pds[r]; else if(idx==2) v=mae[r]; else if(idx==3) v=des[r]; else if(idx==4) v=edist[r];
      else if(idx==5) v=mmd[r]; else if(idx==6) v=r2[r]; else if(idx==7) v=p_all[r]; else if(idx==8) v=pd_all[r];
      else if(idx==9) v=pd20[r]; else if(idx==10) v=pd50[r]; else if(idx==11) v=pd100[r];
      return (v=="") ? 0 : v;
    }
    END {
      printf "==================================================================\n";
      printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
      printf "==================================================================\n";
      for(i=1;i<=11;i++) { ms = mean_std(i); printf "  %d: %s\n", i, ms; }
      metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES"; metric_names[4]="E-Distance"; metric_names[5]="MMD";
      metric_names[6]="R2"; metric_names[7]="Pearson (all genes)"; metric_names[8]="Pearson Delta (all genes)";
      metric_names[9]="Pearson Delta (top 20 DE genes)"; metric_names[10]="Pearson Delta (top 50 DE genes)"; metric_names[11]="Pearson Delta (top 100 DE genes)";
      header="Dataset,Method"; for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)"; for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];
      row=dataset "," method; for(i=1;i<=11;i++) row=row "," mean_std(i); for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);
      print header > csv_path; print row >> csv_path; close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  ' < "${ALL_OUTPUTS_FILE}" || true

  echo
  echo "--- Finished pipeline for MOA: ${moa} ---"
  echo
done

echo "######################################################################"
echo "###   All MOAs processing is complete!                             ###"
echo "######################################################################"