#!/bin/bash

set -e
set -o pipefail

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="scgpt_ddpm"

########################## path ##########################

# PertBench data (relative path for when directory)
TRAIN_H5="data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"

########################## scGPT path ##########################
# under good scGPT checkpoint directory (contain , vocab )
SCGPT_CKPT_DIR="/share/PertBench/checkpoints/scgpt" # TODO: ownpath

# write to PertBench using h5ad (directly X_scgpt latent)
SCGPT_TRAIN_WITH_LATENT="samples/encoder_exp/scgpt_ddpm/task1_train_${CELL_TYPE}_with_scgpt_latent.h5ad"
SCGPT_VALID_WITH_LATENT="samples/encoder_exp/scgpt_ddpm/task1_valid_${CELL_TYPE}_with_scgpt_latent.h5ad"

########################## DDPM & evalpath ##########################

# DDPM checkpoint & eval output ( run subdirin insidedefine)
LATENT_DDPM_CKPT_BASE="checkpoints/scgpt_ddpm/latent_ddpm"
EVAL_OUT_PREFIX="samples/encoder_exp/scgpt_ddpm/scgpt_latent_ddpm_mlp_task1_${CELL_TYPE}_preds"
CSV_PATH="samples/encoder_exp/scgpt_ddpm/metrics_${CELL_TYPE}.csv"

# using DDPM MLP (structuremust ScviLatentDDPMMLP foron)
CONFIG_PATH="configs/baselines/scvi_ddpm_mlp.yaml"

# align gene space afterfile (actual DDPM using)
ALIGNED_TRAIN_H5="samples/encoder_exp/scgpt_ddpm/task1_train_${CELL_TYPE}_scgpt_aligned.h5ad"
ALIGNED_VALID_H5="samples/encoder_exp/scgpt_ddpm/task1_valid_${CELL_TYPE}_scgpt_aligned.h5ad"

LOG_DIR="logs/scgpt_ddpm"

########################## directory ##########################

echo "[INFO] Creating directories for scGPT+DDPM pipeline..."

mkdir -p \
  "samples/encoder_exp/scgpt_ddpm" \
  "checkpoints/scgpt_ddpm" \
  "${LATENT_DDPM_CKPT_BASE}" \
  "${LOG_DIR}"

echo "[INFO] CELL_TYPE                = ${CELL_TYPE}"
echo "[INFO] TRAIN_H5                 = ${TRAIN_H5}"
echo "[INFO] VALID_H5                 = ${VALID_H5}"
echo "[INFO] SCGPT_CKPT_DIR           = ${SCGPT_CKPT_DIR}"
echo "[INFO] SCGPT_TRAIN_WITH_LATENT  = ${SCGPT_TRAIN_WITH_LATENT}"
echo "[INFO] SCGPT_VALID_WITH_LATENT  = ${SCGPT_VALID_WITH_LATENT}"
echo "[INFO] CKPT_BASE                = ${LATENT_DDPM_CKPT_BASE}"
echo "[INFO] CONFIG_PATH              = ${CONFIG_PATH}"
echo

echo "######################################################################"
echo "###   scGPT + latent DDPM pipeline for cell type: ${CELL_TYPE}"
echo "######################################################################"

ALL_OUTPUTS=""

########################################################################
# 0) using scGPT for train/valid , and X_scgpt h5ad (one-time, resume)
########################################################################

echo
echo "======================================================================"
echo ">>> [Stage 0] scGPT encoding for train/valid (${CELL_TYPE})"
echo "======================================================================"

echo -e "\n--- [0.1] Encode TRAIN with scGPT (once for all runs) ---"
if [ -f "${SCGPT_TRAIN_WITH_LATENT}" ]; then
  echo "[Stage 0][TRAIN] Found ${SCGPT_TRAIN_WITH_LATENT}, assuming X_scgpt already exists."
  echo "[Stage 0][TRAIN] Skipping scGPT encoding for TRAIN."
else
  echo "[Stage 0][TRAIN] No ${SCGPT_TRAIN_WITH_LATENT} found. Running scGPT encoder..."
  echo "  python scripts/encoder_exp/scgpt/apply_scgpt_encoder.py \\"
  echo "         --data-path \"${TRAIN_H5}\" \\"
  echo "         --out-h5ad \"${SCGPT_TRAIN_WITH_LATENT}\" \\"
  echo "         --ckpt-dir \"${SCGPT_CKPT_DIR}\" \\"
  echo "         --device cuda"
  python scripts/encoder_exp/scgpt/apply_scgpt_encoder.py \
    --data-path "${TRAIN_H5}" \
    --out-h5ad "${SCGPT_TRAIN_WITH_LATENT}" \
    --ckpt-dir "${SCGPT_CKPT_DIR}" \
    --device cuda 2>&1 | tee "${LOG_DIR}/encode_train_scgpt_${CELL_TYPE}.log"
  echo "[Stage 0][TRAIN] scGPT encoding done. Output: ${SCGPT_TRAIN_WITH_LATENT}"
fi

echo -e "\n--- [0.2] Encode VALID with scGPT (once for all runs) ---"
if [ -f "${SCGPT_VALID_WITH_LATENT}" ]; then
  echo "[Stage 0][VALID] Found ${SCGPT_VALID_WITH_LATENT}, assuming X_scgpt already exists."
  echo "[Stage 0][VALID] Skipping scGPT encoding for VALID."
else
  echo "[Stage 0][VALID] No ${SCGPT_VALID_WITH_LATENT} found. Running scGPT encoder..."
  echo "  python scripts/encoder_exp/scgpt/apply_scgpt_encoder.py \\"
  echo "         --data-path \"${VALID_H5}\" \\"
  echo "         --out-h5ad \"${SCGPT_VALID_WITH_LATENT}\" \\"
  echo "         --ckpt-dir \"${SCGPT_CKPT_DIR}\" \\"
  echo "         --device cuda"
  python scripts/encoder_exp/scgpt/apply_scgpt_encoder.py \
    --data-path "${VALID_H5}" \
    --out-h5ad "${SCGPT_VALID_WITH_LATENT}" \
    --ckpt-dir "${SCGPT_CKPT_DIR}" \
    --device cuda 2>&1 | tee "${LOG_DIR}/encode_valid_scgpt_${CELL_TYPE}.log"
  echo "[Stage 0][VALID] scGPT encoding done. Output: ${SCGPT_VALID_WITH_LATENT}"
fi

########################################################################
# 0.5) align train/valid  gene space
########################################################################

echo
echo "======================================================================"
echo ">>> [Stage 0.5] Align gene space between scGPT-train and scGPT-valid"
echo "======================================================================"

if [ -f "${ALIGNED_TRAIN_H5}" ] && [ -f "${ALIGNED_VALID_H5}" ]; then
  echo "[Stage 0.5] Found aligned h5ad:"
  echo "  TRAIN: ${ALIGNED_TRAIN_H5}"
  echo "  VALID: ${ALIGNED_VALID_H5}"
  echo "[Stage 0.5] Skipping alignment."
else
  echo "[Stage 0.5] Running alignment script..."
  python scripts/encoder_exp/scgpt/align_scgpt_gene_space.py \
    --train-in "${SCGPT_TRAIN_WITH_LATENT}" \
    --valid-in "${SCGPT_VALID_WITH_LATENT}" \
    --train-out "${ALIGNED_TRAIN_H5}" \
    --valid-out "${ALIGNED_VALID_H5}" 2>&1 | tee "${LOG_DIR}/align_scgpt_${CELL_TYPE}.log"
fi

########################################################################
# 1) multi-run: each run separate DDPM checkpoint subdir + evaloutput
########################################################################

for (( run=1; run<=NUM_RUNS; run++ )); do
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  # as run definesubdir
  RUN_CKPT_DIR="${LATENT_DDPM_CKPT_BASE}/run_${run}"
  mkdir -p "${RUN_CKPT_DIR}"
  echo "[RUN ${run}] RUN_CKPT_DIR = ${RUN_CKPT_DIR}"

  ########################################
  # 1) train latent DDPM+decoder (each run ownsubdir, support resume)
  ########################################
  echo -e "\n--- [1/2] Train scGPT-latent DDPM+decoder for ${CELL_TYPE} (run ${run}) ---"

  CKPT_FINAL="${RUN_CKPT_DIR}/model_final.pth"
  echo "[RUN ${run}][DDPM] train-data-path : ${SCGPT_TRAIN_WITH_LATENT}"
  echo "[RUN ${run}][DDPM] ckpt-dir        : ${RUN_CKPT_DIR}"
  echo "[RUN ${run}][DDPM] final-ckpt      : ${CKPT_FINAL}"

  if [ -f "${CKPT_FINAL}" ]; then
    echo "[RUN ${run}][DDPM] Found existing final checkpoint: ${CKPT_FINAL}"
    echo "[RUN ${run}][DDPM] Skip training, reuse this model."
  else
    echo "[RUN ${run}][DDPM] No final checkpoint found. Training (with internal resume if epoch ckpt exists)..."
    echo "  python scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py \\"
    echo "         -c \"${CONFIG_PATH}\" \\"
    echo "         --train-data-path \"${SCGPT_TRAIN_WITH_LATENT}\" \\"
    echo "         --latent-key \"X_scgpt\" \\"
    echo "         --save-weight-dir \"${RUN_CKPT_DIR}\""

    python scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py \
      -c "${CONFIG_PATH}" \
      --train-data-path "${ALIGNED_TRAIN_H5}" \
      --latent-key "X_scgpt" \
      --save-weight-dir "${RUN_CKPT_DIR}" 2>&1 | tee "${LOG_DIR}/train_scgpt_latent_ddpm_${CELL_TYPE}_run_${run}.log"

    echo "[RUN ${run}][DDPM] Training finished (or internally skipped to final)."
  fi

  ########################################
  # 2) eval
  ########################################
  echo -e "\n--- [2/2] Evaluate scGPT-latent model on valid_${CELL_TYPE} (run ${run}) ---"

  CKPT_PATH="${RUN_CKPT_DIR}/model_final.pth"
  if [ ! -f "${CKPT_PATH}" ]; then
    echo "[RUN ${run}][EVAL] WARNING: ${CKPT_PATH} not found, trying to use latest model_epoch_*.pth instead."
    CKPT_PATH=$(ls -1 "${RUN_CKPT_DIR}"/model_epoch_*.pth 2>/dev/null | sort | tail -n 1 || echo "")
  fi

  if [ -z "${CKPT_PATH}" ]; then
    echo "[RUN ${run}][EVAL] ERROR: No checkpoint found in ${RUN_CKPT_DIR}, skipping evaluation for run ${run}."
    continue
  fi

  echo "[RUN ${run}][EVAL] Using checkpoint: ${CKPT_PATH}"
  echo "[RUN ${run}][EVAL] Eval h5ad        : ${SCGPT_VALID_WITH_LATENT}"

  OUTPUT=$(python scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${ALIGNED_VALID_H5}" \
    --latent-key "X_scgpt" \
    -n 200 \
    -o "${EVAL_OUT_PREFIX}_run_${run}.h5ad" 2>&1) || true

  echo "${OUTPUT}"
  ALL_OUTPUTS+="${OUTPUT}"$'\n'
done

########################## metrics to CSV ( scFoundation ) ##########################

echo -e "\n--- Aggregating metrics to CSV: ${CSV_PATH} ---\n"

echo "${ALL_OUTPUTS}" | awk -v dataset="${CELL_TYPE}" -v num_runs="${NUM_RUNS}" \
                       -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
  # here pattern must eval_latent_ddpm_mlp_generic.py output
  /^PDS:/                               { pds[c_pds++] = $NF }
  /^MAE:/                               { mae[c_mae++] = $NF }
  /^DES:/                               { des[c_des++] = $NF }
  /^E-distance:/                        { edist[c_edist++] = $NF }
  /^MMD:/                               { mmd[c_mmd++] = $NF }
  /^R2:/                                { r2[c_r2++] = $NF }
  /^Pearson\(all genes\):/              { pearson_all[c_pearson_all++] = $NF }
  /^Pearson Delta\(all genes\):/        { pearson_delta_all[c_pearson_delta_all++] = $NF }
  /^Pearson Delta\(top 20 DE genes\):/  { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
  /^Pearson Delta\(top 50 DE genes\):/  { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
  /^Pearson Delta\(top 100 DE genes\):/ { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

  function print_stat(name, arr, cnt,    i,sum,mean,ssd,sd){
    if (cnt > 0){
      sum=0
      for(i=0;i<cnt;i++) sum+=arr[i]
      mean=sum/cnt
      ssd=0
      for(i=0;i<cnt;i++) ssd+=(arr[i]-mean)^2
      sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
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

    if(cnt>0){
      mean=sum/cnt; ssd=0
      if(idx==1){ for(i=0;i<cnt;i++) ssd+=(pds[i]-mean)^2 }
      else if(idx==2){ for(i=0;i<cnt;i++) ssd+=(mae[i]-mean)^2 }
      else if(idx==3){ for(i=0;i<cnt;i++) ssd+=(des[i]-mean)^2 }
      else if(idx==4){ for(i=0;i<cnt;i++) ssd+=(edist[i]-mean)^2 }
      else if(idx==5){ for(i=0;i<cnt;i++) ssd+=(mmd[i]-mean)^2 }
      else if(idx==6){ for(i=0;i<cnt;i++) ssd+=(r2[i]-mean)^2 }
      else if(idx==7){ for(i=0;i<cnt;i++) ssd+=(pearson_all[i]-mean)^2 }
      else if(idx==8){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_all[i]-mean)^2 }
      else if(idx==9){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de20[i]-mean)^2 }
      else if(idx==10){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de50[i]-mean)^2 }
      else if(idx==11){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de100[i]-mean)^2 }
      sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
      return sprintf("%.4f|%.4f", mean, sd)
    }
    return "0.0000|0.0000"
  }

  function val_idx(idx, r,    v){
    if     (idx==1){ v = (r < c_pds)?pds[r]:"" }
    else if(idx==2){ v = (r < c_mae)?mae[r]:"" }
    else if(idx==3){ v = (r < c_des)?des[r]:"" }
    else if(idx==4){ v = (r < c_edist)?edist[r]:"" }
    else if(idx==5){ v = (r < c_mmd)?mmd[r]:"" }
    else if(idx==6){ v = (r < c_r2)?r2[r]:"" }
    else if(idx==7){ v = (r < c_pearson_all)?pearson_all[r]:"" }
    else if(idx==8){ v = (r < c_pearson_delta_all)?pearson_delta_all[r]:"" }
    else if(idx==9){ v = (r < c_pearson_delta_de20)?pearson_delta_de20[r]:"" }
    else if(idx==10){ v = (r < c_pearson_delta_de50)?pearson_delta_de50[r]:"" }
    else if(idx==11){ v = (r < c_pearson_delta_de100)?pearson_delta_de100[r]:"" }
    return v
  }

  END{
    print "=================================================================="
    printf " Final statistics for %s (%d runs)\n", dataset, num_runs
    print "=================================================================="

    print_stat("Perturbation Discrimination (PDS)", pds, c_pds)
    print_stat("Mean Absolute Error (MAE)", mae, c_mae)
    print_stat("Differential Expression Score (DES)", des, c_des)
    print "----------------------------------------"
    print_stat("E-Distance", edist, c_edist)
    print_stat("Maximum Mean Discrepancy (MMD)", mmd, c_mmd)
    print_stat("R-squared (R2)", r2, c_r2)
    print "----------------------------------------"
    print_stat("Pearson (all genes)", pearson_all, c_pearson_all)
    print_stat("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all)
    print_stat("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20)
    print_stat("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50)
    print_stat("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100)

    print "==================================================================\n"

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
echo "###   scGPT + latent DDPM pipeline for ${CELL_TYPE} complete!      ###"
echo "######################################################################"
