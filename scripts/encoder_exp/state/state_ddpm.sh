#!/bin/bash
#
# State + DDPM encoder-exp pipeline
# Pipeline: before scRNA -> State SE encoder -> DDPM -> after scRNA
#
# Prerequisites:
#   1. Install State: uv tool install arc-state
#   2. Download SE-600M from HuggingFace (arcinstitute/SE-600M) to STATE_MODEL_DIR
#   3. Set STATE_MODEL_DIR and optionally STATE_CHECKPOINT below
#
# Usage: cd /share/PertBench && bash scripts/encoder_exp/state/state_ddpm.sh
#

set -e

source "scripts/lib/max_n_samples.sh"
set -o pipefail

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="state_ddpm"

########################## State SE path ##########################
STATE_MODEL_DIR="${STATE_MODEL_DIR:-/share/PertBench/checkpoints/SE-600M}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# state (uv tool install to ~/.local/bin)
if command -v state >/dev/null 2>&1; then
  STATE_CMD="state"
elif [ -f "$HOME/.local/bin/state" ]; then
  STATE_CMD="$HOME/.local/bin/state"
  export PATH="$HOME/.local/bin:$PATH"
elif [ -f "$HOME/.cargo/bin/state" ]; then
  STATE_CMD="$HOME/.cargo/bin/state"
  export PATH="$HOME/.cargo/bin:$PATH"
else
  echo "[ERROR] 'state' command not found. Please ensure 'uv tool install arc-state' completed successfully."
  echo "[ERROR] Tried: state, $HOME/.local/bin/state, $HOME/.cargo/bin/state"
  echo "[ERROR] You can set STATE_CMD environment variable to specify the full path."
  exit 1
fi
STATE_CMD="${STATE_CMD:-state}"
echo "[INFO] Using State CLI: $STATE_CMD"

# under SE-600M: directory .ckpt skip; elseunder , whenusing --force-download 
_se_ckpt_exists() {
  [ -d "$1" ] && ls "$1"/*.ckpt 1>/dev/null 2>&1
}
if ! _se_ckpt_exists "$STATE_MODEL_DIR"; then
  echo "[INFO] No SE-600M checkpoint found. Downloading..."
  mkdir -p "$STATE_MODEL_DIR"
  if ! huggingface-cli download arcInstitute/SE-600M --local-dir "$STATE_MODEL_DIR" --resume-download; then
    echo "[WARN] Download failed or incomplete. Retrying with --force-download..."
    huggingface-cli download arcInstitute/SE-600M --local-dir "$STATE_MODEL_DIR" --force-download
  fi
else
  echo "[INFO] SE-600M checkpoint already exists in $STATE_MODEL_DIR. Skipping download."
fi

########################## path ##########################

TRAIN_H5="data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"

# SE-600M path (on STATE_MODEL_DIR)
STATE_CHECKPOINT="${STATE_CHECKPOINT:-}" # empty from model dir .ckpt

# after h5ad ( X_state latent)
STATE_TRAIN_WITH_LATENT="samples/encoder_exp/state_ddpm/task1_train_${CELL_TYPE}_with_state_latent.h5ad"
STATE_VALID_WITH_LATENT="samples/encoder_exp/state_ddpm/task1_valid_${CELL_TYPE}_with_state_latent.h5ad"

########################## DDPM & evalpath ##########################

LATENT_DDPM_CKPT_BASE="checkpoints/state_ddpm/latent_ddpm"
EVAL_OUT_PREFIX="samples/encoder_exp/state_ddpm/state_latent_ddpm_mlp_task1_${CELL_TYPE}_preds"
CSV_PATH="samples/encoder_exp/state_ddpm/metrics_${CELL_TYPE}.csv"

CONFIG_PATH="configs/baselines/state_ddpm_mlp.yaml"
LOG_DIR="logs/state_ddpm"

########################## directory ##########################

echo "[INFO] Creating directories for State+DDPM pipeline..."
mkdir -p \
  "samples/encoder_exp/state_ddpm" \
  "checkpoints/state_ddpm" \
  "${LATENT_DDPM_CKPT_BASE}" \
  "${LOG_DIR}"

echo "[INFO] CELL_TYPE                = ${CELL_TYPE}"
echo "[INFO] TRAIN_H5                 = ${TRAIN_H5}"
echo "[INFO] VALID_H5                 = ${VALID_H5}"
echo "[INFO] STATE_MODEL_DIR          = ${STATE_MODEL_DIR}"
echo "[INFO] STATE_TRAIN_WITH_LATENT  = ${STATE_TRAIN_WITH_LATENT}"
echo "[INFO] STATE_VALID_WITH_LATENT  = ${STATE_VALID_WITH_LATENT}"
echo "[INFO] CKPT_BASE                = ${LATENT_DDPM_CKPT_BASE}"
echo "[INFO] CONFIG_PATH              = ${CONFIG_PATH}"
echo

echo "######################################################################"
echo "###   State SE + latent DDPM pipeline for cell type: ${CELL_TYPE}"
echo "######################################################################"

ALL_OUTPUTS=""

########################################################################
# Stage 0) State SE train/valid (one-time, support resume)
########################################################################

echo
echo "======================================================================"
echo ">>> [Stage 0] State SE encoding for train/valid (${CELL_TYPE})"
echo "======================================================================"

BATCH_CELLS="${BATCH_CELLS:-500}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
# Use State's Python interpreter to avoid dependency conflicts
STATE_PYTHON="${HOME}/.local/share/uv/tools/arc-state/bin/python"
if [ ! -f "${STATE_PYTHON}" ]; then
  echo "[WARN] State Python not found at ${STATE_PYTHON}, using system python"
  STATE_PYTHON="python"
fi
encode_train_cmd="${STATE_PYTHON} scripts/encoder_exp/state/apply_state_encoder.py \
  --data-path \"${TRAIN_H5}\" \
  --out-h5ad \"${STATE_TRAIN_WITH_LATENT}\" \
  --model-folder \"${STATE_MODEL_DIR}\" \
  --batch-cells ${BATCH_CELLS} \
  --embed-batch-size ${EMBED_BATCH_SIZE}"
[ -n "${STATE_CHECKPOINT}" ] && encode_train_cmd="${encode_train_cmd} --checkpoint \"${STATE_CHECKPOINT}\""

echo -e "\n--- [0.1] Encode TRAIN with State SE ---"
if [ -f "${STATE_TRAIN_WITH_LATENT}" ]; then
  echo "[Stage 0][TRAIN] Found ${STATE_TRAIN_WITH_LATENT}, assuming X_state exists. Skipping."
else
  echo "[Stage 0][TRAIN] Running State encoder..."
  eval "${encode_train_cmd}" 2>&1 | tee "${LOG_DIR}/encode_train_state_${CELL_TYPE}.log"
  echo "[Stage 0][TRAIN] Done. Output: ${STATE_TRAIN_WITH_LATENT}"
fi

encode_valid_cmd="${STATE_PYTHON} scripts/encoder_exp/state/apply_state_encoder.py \
  --data-path \"${VALID_H5}\" \
  --out-h5ad \"${STATE_VALID_WITH_LATENT}\" \
  --model-folder \"${STATE_MODEL_DIR}\" \
  --batch-cells ${BATCH_CELLS} \
  --embed-batch-size ${EMBED_BATCH_SIZE}"
[ -n "${STATE_CHECKPOINT}" ] && encode_valid_cmd="${encode_valid_cmd} --checkpoint \"${STATE_CHECKPOINT}\""

echo -e "\n--- [0.2] Encode VALID with State SE ---"
if [ -f "${STATE_VALID_WITH_LATENT}" ]; then
  echo "[Stage 0][VALID] Found ${STATE_VALID_WITH_LATENT}, assuming X_state exists. Skipping."
else
  echo "[Stage 0][VALID] Running State encoder..."
  eval "${encode_valid_cmd}" 2>&1 | tee "${LOG_DIR}/encode_valid_state_${CELL_TYPE}.log"
  echo "[Stage 0][VALID] Done. Output: ${STATE_VALID_WITH_LATENT}"
fi

########################################################################
# Stage 1) multi-run: DDPM train + eval
########################################################################

for (( run=1; run<=NUM_RUNS; run++ )); do
  export RUN_SEED=$(($run-1))
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  RUN_CKPT_DIR="${LATENT_DDPM_CKPT_BASE}/run_${run}"
  mkdir -p "${RUN_CKPT_DIR}"
  echo "[RUN ${run}] RUN_CKPT_DIR = ${RUN_CKPT_DIR}"

  ########################################
  # 1) train latent DDPM+decoder
  ########################################
  echo -e "\n--- [1/2] Train State-latent DDPM+decoder for ${CELL_TYPE} (run ${run}) ---"

  CKPT_FINAL="${RUN_CKPT_DIR}/model_final.pth"
  if [ -f "${CKPT_FINAL}" ]; then
    echo "[RUN ${run}][DDPM] Found ${CKPT_FINAL}. Skip training."
  else
    echo "[RUN ${run}][DDPM] Training..."
    python scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py \
      -c "${CONFIG_PATH}" \
      --train-data-path "${STATE_TRAIN_WITH_LATENT}" \
      --latent-key "X_state" \
      --save-weight-dir "${RUN_CKPT_DIR}" 2>&1 | tee "${LOG_DIR}/train_state_latent_ddpm_${CELL_TYPE}_run_${run}.log"
    echo "[RUN ${run}][DDPM] Training finished."
  fi

  ########################################
  # 2) eval
  ########################################
  echo -e "\n--- [2/2] Evaluate State-latent model on valid_${CELL_TYPE} (run ${run}) ---"

  CKPT_PATH="${RUN_CKPT_DIR}/model_final.pth"
  if [ ! -f "${CKPT_PATH}" ]; then
    CKPT_PATH=$(ls -1 "${RUN_CKPT_DIR}"/model_epoch_*.pth 2>/dev/null | sort | tail -n 1 || echo "")
  fi

  if [ -z "${CKPT_PATH}" ] || [ ! -f "${CKPT_PATH}" ]; then
    echo "[RUN ${run}][EVAL] ERROR: No checkpoint in ${RUN_CKPT_DIR}. Skipping."
    continue
  fi

  echo "[RUN ${run}][EVAL] Using checkpoint: ${CKPT_PATH}"
  OUTPUT=$(python scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${STATE_VALID_WITH_LATENT}" \
    --latent-key "X_state" \
    -n 200 \
    -o "${EVAL_OUT_PREFIX}_run_${run}.h5ad" 2>&1) || true

  echo "${OUTPUT}"
  ALL_OUTPUTS+="${OUTPUT}"$'\n'
done

########################## metrics to CSV ##########################

echo -e "\n--- Aggregating metrics to CSV: ${CSV_PATH} ---\n"

echo "${ALL_OUTPUTS}" | awk -v dataset="${CELL_TYPE}" -v num_runs="${NUM_RUNS}" \
                       -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
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
echo "###   State + DDPM pipeline for ${CELL_TYPE} complete!               ###"
echo "######################################################################"
