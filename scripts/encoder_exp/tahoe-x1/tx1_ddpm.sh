#!/bin/bash
set -e

export TX1_MODEL_SIZE=70m
export HF_ENDPOINT=https://hf-mirror.com

ROOT_DIR="$(pwd)"

# Hard fail if not in repo root (prevents "cd to wrong place" bugs)
if [[ ! -d "${ROOT_DIR}/src" ]] || [[ ! -d "${ROOT_DIR}/utils" ]] || [[ ! -d "${ROOT_DIR}/scripts" ]]; then
  echo "[FATAL] Please run this script from PertBench repo root. Current dir: ${ROOT_DIR}" >&2
  echo "Example: cd /share/PertBench && nohup bash scripts/encoder_exp/tahoe-x1/tx1_ddpm.sh ..." >&2
  exit 1
fi

# Make python imports robust
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src/tahoe-x1:${PYTHONPATH}"

# Help apply_tx1_encoder.py locate repo script fallback
export TAHOEX1_ROOT="${ROOT_DIR}/src/tahoe-x1"

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="tx1_ddpm"

# -------- Tx1 encoder settings --------
# 若从 Hugging Face 下载 3b 失败（如网络/墙），可改用小模型： export TX1_MODEL_SIZE=70m 后再运行
TX1_REPO_ID="tahoebio/Tahoe-x1"
TX1_MODEL_SIZE="${TX1_MODEL_SIZE:-3b}"   # 70m | 1b | 3b，可用环境变量覆盖
TX1_OBSM_KEY="X_tx1"
TX1_SEQ_LEN=2048

# -------- Data paths (align with your existing scVI pipeline) --------
TRAIN_H5="data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"

TRAIN_H5_WITH_LATENT="samples/encoder_exp/tx1_ddpm/task1_train_${CELL_TYPE}_with_tx1_latent.h5ad"
VALID_H5_WITH_LATENT="samples/encoder_exp/tx1_ddpm/task1_valid_${CELL_TYPE}_exp_with_tx1_latent.h5ad"

CONFIG_PATH="configs/baselines/encoder_tahoex1_ddpm.yaml"
LOG_DIR="logs/tx1_ddpm/${CELL_TYPE}"
CSV_PATH="samples/encoder_exp/tx1_ddpm/tx1_ddpm_${CELL_TYPE}.csv"
EVAL_OUT_PREFIX="samples/encoder_exp/tx1_ddpm/tx1_ddpm_synth_${CELL_TYPE}"

mkdir -p \
  "samples/encoder_exp/tx1_ddpm" \
  "checkpoints/tx1_ddpm" \
  "${LOG_DIR}" \
  "outputs"

echo "######################################################################"
echo "###   Tx1+DDPM pipeline for cell type: ${CELL_TYPE}"
echo "###   Tx1: ${TX1_REPO_ID} (${TX1_MODEL_SIZE})"
echo "######################################################################"

########################################
# 0) Tx1 encode train + valid (once; apply_tx1_encoder.py skips if obsm already present)
########################################
echo -e "\n--- [0/3] Tx1 encode train & valid (once, skip if output exists) ---"
if [ -f "${TRAIN_H5_WITH_LATENT}" ] && [ -f "${VALID_H5_WITH_LATENT}" ]; then
  echo "[Tx1] Found existing encoded h5ad, skip encoding. Delete to re-encode:"
  echo "  ${TRAIN_H5_WITH_LATENT}"
  echo "  ${VALID_H5_WITH_LATENT}"
else
  echo "[Tx1] Encoding TRAIN -> ${TRAIN_H5_WITH_LATENT}"
  python scripts/encoder_exp/tahoe-x1/apply_tx1_encoder.py \
    --data-path "${TRAIN_H5}" \
    --out-h5ad "${TRAIN_H5_WITH_LATENT}" \
    --hf-repo-id "${TX1_REPO_ID}" \
    --model-size "${TX1_MODEL_SIZE}" \
    --obsm-key "${TX1_OBSM_KEY}" \
    --seq-len-dataset "${TX1_SEQ_LEN}" \
    --gpu 2>&1 | tee "${LOG_DIR}/encode_train_${CELL_TYPE}.log"

  echo "[Tx1] Encoding VALID -> ${VALID_H5_WITH_LATENT}"
  python scripts/encoder_exp/tahoe-x1/apply_tx1_encoder.py \
    --data-path "${VALID_H5}" \
    --out-h5ad "${VALID_H5_WITH_LATENT}" \
    --hf-repo-id "${TX1_REPO_ID}" \
    --model-size "${TX1_MODEL_SIZE}" \
    --obsm-key "${TX1_OBSM_KEY}" \
    --seq-len-dataset "${TX1_SEQ_LEN}" \
    --gpu 2>&1 | tee "${LOG_DIR}/encode_valid_${CELL_TYPE}.log"
fi

ALL_OUTPUTS=""

for (( run=1; run<=NUM_RUNS; run++ )); do
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  LATENT_DDPM_CKPT_DIR="checkpoints/tx1_ddpm/latent_ddpm_run_${run}"

  ########################################
  # 1) Train latent DDPM+decoder (auto-resume; skip if model_final.pth exists)
  ########################################
  echo -e "\n--- [1/3] Train latent DDPM-MLP (run ${run}) ---"
  python scripts/encoder_exp/tahoe-x1/train_tx1_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    --train-data-path "${TRAIN_H5_WITH_LATENT}" \
    --obsm-key "${TX1_OBSM_KEY}" \
    --save-weight-dir "${LATENT_DDPM_CKPT_DIR}" \
    --resume auto 2>&1 | tee "${LOG_DIR}/train_latent_ddpm_${CELL_TYPE}_run_${run}.log"

  ########################################
  # 2) Evaluate (collect stdout for CSV)
  ########################################
  echo -e "\n--- [2/3] Evaluate on valid set (run ${run}) ---"
  CKPT_PATH="${LATENT_DDPM_CKPT_DIR}/model_final.pth"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "[FATAL] Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
  fi

  OUTPUT=$(python scripts/encoder_exp/tahoe-x1/eval_tx1_encoder_compat.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${VALID_H5_WITH_LATENT}" \
    --obsm-key "${TX1_OBSM_KEY}" \
    -n 200 \
    -o "${EVAL_OUT_PREFIX}_run_${run}.h5ad" 2>&1) || true

  echo "${OUTPUT}"
  ALL_OUTPUTS+="${OUTPUT}"$'\n'
done

echo -e "\n--- Aggregating metrics to CSV: ${CSV_PATH} ---\n"

echo "${ALL_OUTPUTS}" | awk -v dataset="${CELL_TYPE}" -v num_runs="${NUM_RUNS}" \
                       -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
  /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
  /Mean Absolute Error \(MAE\):/             { mae[c_mae++] = $NF }
  /Differential Expression Score \(DES\):/   { des[c_des++] = $NF }
  /E-Distance:/                              { edist[c_edist++] = $NF }
  /Maximum Mean Discrepancy \(MMD\):/        { mmd[c_mmd++] = $NF }
  /R-squared \(R2\):/                        { r2[c_r2++] = $NF }
  /Pearson \(all genes\):/                   { pearson_all[c_pearson_all++] = $NF }
  /Pearson Delta \(all genes\):/             { pearson_delta_all[c_pearson_delta_all++] = $NF }
  /Pearson Delta \(top 20 DE genes\):/       { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
  /Pearson Delta \(top 50 DE genes\):/       { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
  /Pearson Delta \(top 100 DE genes\):/      { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

  function print_mean_std(name, arr, n,    i,sum,mean,ssd,sd){
    if(n>0){
      sum=0
      for(i=0;i<n;i++) sum+=arr[i]
      mean = sum/n
      ssd=0
      for(i=0;i<n;i++) ssd += (arr[i]-mean)*(arr[i]-mean)
      sd=(n>1)?sqrt(ssd/(n-1)):0
      printf "%-40s: %.4f ± %.4f\n", name, mean, sd
    } else {
      printf "%-40s: N/A (No data collected)\n", name
    }
  }

  function mean_std_str(idx,    i,sum,mean,ssd,sd,cnt){
    if(idx==1){ cnt=c_pds; for(i=0;i<cnt;i++) sum+=pds[i] }
    else if(idx==2){ cnt=c_mae; for(i=0;i<cnt;i++) sum+=mae[i] }
    else if(idx==3){ cnt=c_des; for(i=0;i<cnt;i++) sum+=des[i] }
    else if(idx==4){ cnt=c_edist; for(i=0;i<cnt;i++) sum+=edist[i] }
    else if(idx==5){ cnt=c_mmd; for(i=0;i<cnt;i++) sum+=mmd[i] }
    else if(idx==6){ cnt=c_r2; for(i=0;i<cnt;i++) sum+=r2[i] }
    else if(idx==7){ cnt=c_pearson_all; for(i=0;i<cnt;i++) sum+=pearson_all[i] }
    else if(idx==8){ cnt=c_pearson_delta_all; for(i=0;i<cnt;i++) sum+=pearson_delta_all[i] }
    else if(idx==9){ cnt=c_pearson_delta_de20; for(i=0;i<cnt;i++) sum+=pearson_delta_de20[i] }
    else if(idx==10){ cnt=c_pearson_delta_de50; for(i=0;i<cnt;i++) sum+=pearson_delta_de50[i] }
    else if(idx==11){ cnt=c_pearson_delta_de100; for(i=0;i<cnt;i++) sum+=pearson_delta_de100[i] }

    mean = (cnt>0)?(sum/cnt):0
    ssd=0
    for(i=0;i<cnt;i++){
      if(idx==1) ssd+=(pds[i]-mean)*(pds[i]-mean)
      else if(idx==2) ssd+=(mae[i]-mean)*(mae[i]-mean)
      else if(idx==3) ssd+=(des[i]-mean)*(des[i]-mean)
      else if(idx==4) ssd+=(edist[i]-mean)*(edist[i]-mean)
      else if(idx==5) ssd+=(mmd[i]-mean)*(mmd[i]-mean)
      else if(idx==6) ssd+=(r2[i]-mean)*(r2[i]-mean)
      else if(idx==7) ssd+=(pearson_all[i]-mean)*(pearson_all[i]-mean)
      else if(idx==8) ssd+=(pearson_delta_all[i]-mean)*(pearson_delta_all[i]-mean)
      else if(idx==9) ssd+=(pearson_delta_de20[i]-mean)*(pearson_delta_de20[i]-mean)
      else if(idx==10) ssd+=(pearson_delta_de50[i]-mean)*(pearson_delta_de50[i]-mean)
      else if(idx==11) ssd+=(pearson_delta_de100[i]-mean)*(pearson_delta_de100[i]-mean)
    }
    sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
    return mean "|" sd
  }

  function val_idx(idx, run_idx){
    if(idx==1) return pds[run_idx]
    else if(idx==2) return mae[run_idx]
    else if(idx==3) return des[run_idx]
    else if(idx==4) return edist[run_idx]
    else if(idx==5) return mmd[run_idx]
    else if(idx==6) return r2[run_idx]
    else if(idx==7) return pearson_all[run_idx]
    else if(idx==8) return pearson_delta_all[run_idx]
    else if(idx==9) return pearson_delta_de20[run_idx]
    else if(idx==10) return pearson_delta_de50[run_idx]
    else if(idx==11) return pearson_delta_de100[run_idx]
    else return ""
  }

  END {
    print_mean_std("Perturbation Discrimination Score (PDS)", pds, c_pds)
    print_mean_std("Mean Absolute Error (MAE)", mae, c_mae)
    print_mean_std("Differential Expression Score (DES)", des, c_des)
    print_mean_std("E-Distance", edist, c_edist)
    print_mean_std("Maximum Mean Discrepancy (MMD)", mmd, c_mmd)
    print_mean_std("R-squared (R2)", r2, c_r2)
    print_mean_std("Pearson (all genes)", pearson_all, c_pearson_all)
    print_mean_std("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all)
    print_mean_std("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20)
    print_mean_std("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50)
    print_mean_std("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100)

    metric_names[1]="PDS"
    metric_names[2]="MAE"
    metric_names[3]="DES"
    metric_names[4]="E-Distance"
    metric_names[5]="MMD"
    metric_names[6]="R2"
    metric_names[7]="Pearson (all genes)"
    metric_names[8]="Pearson Delta (all genes)"
    metric_names[9]="Pearson Delta (top 20 DE genes)"
    metric_names[10]="Pearson Delta (top 50 DE genes)"
    metric_names[11]="Pearson Delta (top 100 DE genes)"

    header = "Method"
    for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)"
    for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i]

    row = method
    for (i=1;i<=11;i++){
      ms = mean_std_str(i)
      split(ms, parts, "|")
      row = row sprintf(",%.4f±%.4f", parts[1], parts[2])
    }
    for (r=0;r<num_runs;r++){
      for (i=1;i<=11;i++){
        v = val_idx(i, r)
        if (v == "") row = row ","
        else row = row sprintf(",%.4f", v)
      }
    }

    print header > csv_path
    print row    >> csv_path
    close(csv_path)
    printf("CSV written: %s\n", csv_path)
  }
'

echo "######################################################################"
echo "###   Tx1+DDPM pipeline for ${CELL_TYPE} complete!                  ###"
echo "######################################################################"
