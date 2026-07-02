#!/bin/bash

set -e

source "scripts/lib/max_n_samples.sh"

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="scfoundation_ddpm"

########################## path ##########################

# PertBench data (relative path for when directory)
TRAIN_H5="data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"

# scFoundation repo root (change this to your actual path)
SCF_ROOT="/share/PertBench/src/scFoundation"

# scFoundation preprocessing & embedding paths
SCF_PREPROC_DIR="${SCF_ROOT}/preprocessing"
SCF_PREPROC_SCRIPT="${SCF_PREPROC_DIR}/scRNAseq_h5ad_preprocessing_under_scfoundation.py"
SCF_MODEL_DIR="${SCF_ROOT}/model"

# preprocessoutput h5ad ( : preprocess mustwrite to path)
SCF_TRAIN_PRE_H5="${SCF_PREPROC_DIR}/preprocessed_task1_train_${CELL_TYPE}_exp.h5ad"
SCF_VALID_PRE_H5="${SCF_PREPROC_DIR}/preprocessed_task1_valid_${CELL_TYPE}_exp.h5ad"

# get_embedding.py willin model directoryunder 
SCF_GET_EMB="${SCF_MODEL_DIR}/get_embedding.py"
SCF_EXAMPLE_DIR="${SCF_MODEL_DIR}/examples/single_cell_data"
SCF_OUTPUT_DIR="${SCF_MODEL_DIR}/output/single_cell_data"

# we rename .npy, under using
SCF_TRAIN_EMB_NPY="${SCF_OUTPUT_DIR}/task1_train_${CELL_TYPE}_exp_cell_embedding.npy"
SCF_VALID_EMB_NPY="${SCF_OUTPUT_DIR}/task1_valid_${CELL_TYPE}_exp_cell_embedding.npy"

# write to PertBench using h5ad
SCF_TRAIN_WITH_LATENT="samples/encoder_exp/scfoundation_ddpm/task1_train_${CELL_TYPE}_with_scf_latent.h5ad"
SCF_VALID_WITH_LATENT="samples/encoder_exp/scfoundation_ddpm/task1_valid_${CELL_TYPE}_with_scf_latent.h5ad"

# DDPM checkpoint & eval output ( : run subdirin insidedefine)
LATENT_DDPM_CKPT_BASE="checkpoints/scfoundation_ddpm/latent_ddpm"
EVAL_OUT_PREFIX="samples/encoder_exp/scfoundation_ddpm/scf_latent_ddpm_mlp_task1_${CELL_TYPE}_preds"
CSV_PATH="samples/encoder_exp/scfoundation_ddpm/metrics_${CELL_TYPE}.csv"

CONFIG_PATH="configs/baselines/scvi_ddpm_mlp.yaml"
LOG_DIR="logs/scfoundation_ddpm"

########################## directory ##########################

echo "[INFO] Creating directories..."

mkdir -p \
  "samples/encoder_exp/scfoundation_ddpm" \
  "checkpoints/scfoundation_ddpm" \
  "${LATENT_DDPM_CKPT_BASE}" \
  "${LOG_DIR}" \
  "${SCF_EXAMPLE_DIR}" \
  "${SCF_OUTPUT_DIR}" \
  "${SCF_PREPROC_DIR}/data" \
  "${SCF_PREPROC_DIR}/output"

echo "[INFO] CELL_TYPE          = ${CELL_TYPE}"
echo "[INFO] TRAIN_H5           = ${TRAIN_H5}"
echo "[INFO] VALID_H5           = ${VALID_H5}"
echo "[INFO] SCF_ROOT           = ${SCF_ROOT}"
echo "[INFO] SCF_PREPROC_DIR    = ${SCF_PREPROC_DIR}"
echo "[INFO] SCF_MODEL_DIR      = ${SCF_MODEL_DIR}"
echo "[INFO] SCF_TRAIN_PRE_H5   = ${SCF_TRAIN_PRE_H5}"
echo "[INFO] SCF_VALID_PRE_H5   = ${SCF_VALID_PRE_H5}"
echo "[INFO] SCF_TRAIN_EMB_NPY  = ${SCF_TRAIN_EMB_NPY}"
echo "[INFO] SCF_VALID_EMB_NPY  = ${SCF_VALID_EMB_NPY}"
echo "[INFO] CKPT_BASE          = ${LATENT_DDPM_CKPT_BASE}"
echo

echo "######################################################################"
echo "###   scFoundation+DDPM pipeline for cell type: ${CELL_TYPE}"
echo "######################################################################"

ALL_OUTPUTS=""

for (( run=1; run<=NUM_RUNS; run++ )); do
  export RUN_SEED=$(($run-1))
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  # as run definesubdir
  RUN_CKPT_DIR="${LATENT_DDPM_CKPT_BASE}/run_${run}"
  mkdir -p "${RUN_CKPT_DIR}"
  echo "[RUN ${run}] RUN_CKPT_DIR = ${RUN_CKPT_DIR}"

  ########################################
  # 1) scFoundation preprocessing + embedding for TRAIN
  ########################################
  echo -e "\n--- [1/4] scFoundation preprocessing + embedding on train_${CELL_TYPE} (run ${run}) ---"

  echo "[STEP 1.1] Preprocess TRAIN with scFoundation"
  echo "[STEP 1.1] Copying ${TRAIN_H5} -> ${SCF_PREPROC_DIR}/data/task1_train_${CELL_TYPE}_exp.h5ad"

  # copyoriginal train h5ad to scFoundation data directory ( )
  cp "${TRAIN_H5}" "${SCF_PREPROC_DIR}/data/task1_train_${CELL_TYPE}_exp.h5ad"

  if [ -f "${SCF_TRAIN_PRE_H5}" ]; then
    echo "[SCFoundation] Found preprocessed TRAIN h5ad: ${SCF_TRAIN_PRE_H5}"
    echo "[SCFoundation] Skip TRAIN preprocessing."
  else
    echo "[SCFoundation] No preprocessed TRAIN h5ad found."
    echo "[SCFoundation] Running preprocessing script:"
    echo "  python ${SCF_PREPROC_SCRIPT} --system_path ${SCF_PREPROC_DIR} --file_name task1_train_${CELL_TYPE}_exp.h5ad --sparse_matrix False"

    python "${SCF_PREPROC_SCRIPT}" \
      --system_path "${SCF_PREPROC_DIR}" \
      --file_name "task1_train_${CELL_TYPE}_exp.h5ad" \
      --sparse_matrix False 2>&1 | tee "${LOG_DIR}/scf_preproc_train_${CELL_TYPE}_run_${run}.log"

    echo "[SCFoundation] TRAIN preprocessing finished, expected output: ${SCF_TRAIN_PRE_H5}"
  fi

  echo "[STEP 1.2] Prepare preprocessed TRAIN for get_embedding"
  echo "[STEP 1.2] Copying ${SCF_TRAIN_PRE_H5} -> ${SCF_EXAMPLE_DIR}/preprocessed_task1_train_${CELL_TYPE}_exp.h5ad"
  cp "${SCF_TRAIN_PRE_H5}" "${SCF_EXAMPLE_DIR}/preprocessed_task1_train_${CELL_TYPE}_exp.h5ad"

  echo "[STEP 1.3] Compute TRAIN embedding (if not already present)"
  if [ -f "${SCF_TRAIN_EMB_NPY}" ]; then
    echo "[SCFoundation] Found existing TRAIN embedding: ${SCF_TRAIN_EMB_NPY}"
    echo "[SCFoundation] Skip get_embedding for TRAIN."
  else
    echo "[SCFoundation] No existing TRAIN embedding, running get_embedding..."
    echo "  python ${SCF_GET_EMB} --task_name task1_train_${CELL_TYPE} --input_type singlecell --output_type cell --pool_type all"
    echo "         --data_path ${SCF_EXAMPLE_DIR}/preprocessed_task1_train_${CELL_TYPE}_exp.h5ad --pre_normalized F --version rde"
    echo "         --save_path ${SCF_OUTPUT_DIR} --tgthighres f1"

    python "${SCF_GET_EMB}" \
      --task_name "task1_train_${CELL_TYPE}" \
      --input_type singlecell \
      --output_type cell \
      --pool_type all \
      --data_path "${SCF_EXAMPLE_DIR}/preprocessed_task1_train_${CELL_TYPE}_exp.h5ad" \
      --pre_normalized F \
      --version rde \
      --save_path "${SCF_OUTPUT_DIR}" \
      --tgthighres f1 2>&1 | tee "${LOG_DIR}/scf_getemb_train_${CELL_TYPE}_run_${run}.log"

    # : hereusing when tooutputfilename
    RAW_TRAIN_NPY="${SCF_OUTPUT_DIR}/task1_train_CD4T_01B-resolution_singlecell_cell_embedding_f1_resolution.npy"
    echo "[SCFoundation] Moving raw TRAIN embedding:"
    echo "  ${RAW_TRAIN_NPY} -> ${SCF_TRAIN_EMB_NPY}"
    mv "${RAW_TRAIN_NPY}" "${SCF_TRAIN_EMB_NPY}"
  fi

  echo "[STEP 1.4] Attach TRAIN embedding back to PertBench AnnData"
  echo "  orig-h5ad : ${TRAIN_H5}"
  echo "  pre-h5ad  : ${SCF_TRAIN_PRE_H5}"
  echo "  embedding : ${SCF_TRAIN_EMB_NPY}"
  echo "  out-h5ad  : ${SCF_TRAIN_WITH_LATENT}"

  python scripts/encoder_exp/scfoundation/attach_scfoundation_embedding.py \
    --orig-h5ad "${TRAIN_H5}" \
    --pre-h5ad "${SCF_TRAIN_PRE_H5}" \
    --embedding-npy "${SCF_TRAIN_EMB_NPY}" \
    --out-h5ad "${SCF_TRAIN_WITH_LATENT}" \
    --obsm-key "X_scfoundation" 2>&1 | tee "${LOG_DIR}/attach_scf_train_${CELL_TYPE}_run_${run}.log"

  ########################################
  # 2) scFoundation preprocessing + embedding for VALID
  ########################################
  echo -e "\n--- [2/4] scFoundation preprocessing + embedding on valid_${CELL_TYPE} (run ${run}) ---"

  echo "[STEP 2.1] Preprocess VALID with scFoundation"
  echo "[STEP 2.1] Copying ${VALID_H5} -> ${SCF_PREPROC_DIR}/data/task1_valid_${CELL_TYPE}_exp.h5ad"

  cp "${VALID_H5}" "${SCF_PREPROC_DIR}/data/task1_valid_${CELL_TYPE}_exp.h5ad"

  if [ -f "${SCF_VALID_PRE_H5}" ]; then
    echo "[SCFoundation] Found preprocessed VALID h5ad: ${SCF_VALID_PRE_H5}"
    echo "[SCFoundation] Skip VALID preprocessing."
  else
    echo "[SCFoundation] No preprocessed VALID h5ad found."
    echo "[SCFoundation] Running preprocessing script:"
    echo "  python ${SCF_PREPROC_SCRIPT} --system_path ${SCF_PREPROC_DIR} --file_name task1_valid_${CELL_TYPE}_exp.h5ad --sparse_matrix False"

    python "${SCF_PREPROC_SCRIPT}" \
      --system_path "${SCF_PREPROC_DIR}" \
      --file_name "task1_valid_${CELL_TYPE}_exp.h5ad" \
      --sparse_matrix False 2>&1 | tee "${LOG_DIR}/scf_preproc_valid_${CELL_TYPE}_run_${run}.log"

    echo "[SCFoundation] VALID preprocessing finished, expected output: ${SCF_VALID_PRE_H5}"
  fi

  echo "[STEP 2.2] Prepare preprocessed VALID for get_embedding"
  echo "[STEP 2.2] Copying ${SCF_VALID_PRE_H5} -> ${SCF_EXAMPLE_DIR}/preprocessed_task1_valid_${CELL_TYPE}_exp.h5ad"
  cp "${SCF_VALID_PRE_H5}" "${SCF_EXAMPLE_DIR}/preprocessed_task1_valid_${CELL_TYPE}_exp.h5ad"

  echo "[STEP 2.3] Compute VALID embedding (if not already present)"
  if [ -f "${SCF_VALID_EMB_NPY}" ]; then
    echo "[SCFoundation] Found existing VALID embedding: ${SCF_VALID_EMB_NPY}"
    echo "[SCFoundation] Skip get_embedding for VALID."
  else
    echo "[SCFoundation] No existing VALID embedding, running get_embedding..."
    echo "  python ${SCF_GET_EMB} --task_name task1_valid_${CELL_TYPE} --input_type singlecell --output_type cell --pool_type all"
    echo "         --data_path ${SCF_EXAMPLE_DIR}/preprocessed_task1_valid_${CELL_TYPE}_exp.h5ad --pre_normalized F --version rde"
    echo "         --save_path ${SCF_OUTPUT_DIR} --tgthighres f1"

    python "${SCF_GET_EMB}" \
      --task_name "task1_valid_${CELL_TYPE}" \
      --input_type singlecell \
      --output_type cell \
      --pool_type all \
      --data_path "${SCF_EXAMPLE_DIR}/preprocessed_task1_valid_${CELL_TYPE}_exp.h5ad" \
      --pre_normalized F \
      --version rde \
      --save_path "${SCF_OUTPUT_DIR}" \
      --tgthighres f1 2>&1 | tee "${LOG_DIR}/scf_getemb_valid_${CELL_TYPE}_run_${run}.log"

    RAW_VALID_NPY="${SCF_OUTPUT_DIR}/task1_valid_CD4T_01B-resolution_singlecell_cell_embedding_f1_resolution.npy"
    echo "[SCFoundation] Moving raw VALID embedding:"
    echo "  ${RAW_VALID_NPY} -> ${SCF_VALID_EMB_NPY}"
    mv "${RAW_VALID_NPY}" "${SCF_VALID_EMB_NPY}"
  fi

  echo "[STEP 2.4] Attach VALID embedding back to PertBench AnnData"
  echo "  orig-h5ad : ${VALID_H5}"
  echo "  pre-h5ad  : ${SCF_VALID_PRE_H5}"
  echo "  embedding : ${SCF_VALID_EMB_NPY}"
  echo "  out-h5ad  : ${SCF_VALID_WITH_LATENT}"

  python scripts/encoder_exp/scfoundation/attach_scfoundation_embedding.py \
    --orig-h5ad "${VALID_H5}" \
    --pre-h5ad "${SCF_VALID_PRE_H5}" \
    --embedding-npy "${SCF_VALID_EMB_NPY}" \
    --out-h5ad "${SCF_VALID_WITH_LATENT}" \
    --obsm-key "X_scfoundation" 2>&1 | tee "${LOG_DIR}/attach_scf_valid_${CELL_TYPE}_run_${run}.log"

  #########################################
  # 3) train latent DDPM+decoder (each run ownsubdir)
  #########################################
  echo -e "\n--- [3/4] Train scFoundation-latent DDPM+decoder for ${CELL_TYPE} (run ${run}) ---"

  CKPT_PATH="${RUN_CKPT_DIR}/model_final.pth"
  echo "[STEP 3] DDPM training stage (run ${run})"
  echo "  train-data-path : ${SCF_TRAIN_WITH_LATENT}"
  echo "  run-ckpt-dir    : ${RUN_CKPT_DIR}"
  echo "  ckpt-path       : ${CKPT_PATH}"

  if [ -f "${CKPT_PATH}" ]; then
    echo "[DDPM][run ${run}] Found existing checkpoint: ${CKPT_PATH}"
    echo "[DDPM][run ${run}] Skip training, reuse this model."
  else
    echo "[DDPM][run ${run}] No checkpoint found, start training..."
    python scripts/encoder_exp/scfoundation/train_scfoundation_latent_ddpm_mlp.py \
      -c "${CONFIG_PATH}" \
      --train-data-path "${SCF_TRAIN_WITH_LATENT}" \
      --save-weight-dir "${RUN_CKPT_DIR}" \
      --latent-key "X_scfoundation" 2>&1 | tee "${LOG_DIR}/train_scf_latent_ddpm_${CELL_TYPE}_run_${run}.log"
  fi

  ########################################
  # 4) eval
  ########################################
  echo -e "\n--- [4/4] Evaluate scFoundation-latent model on valid_${CELL_TYPE} (run ${run}) ---"

  CKPT_PATH="${RUN_CKPT_DIR}/model_final.pth"
  echo "[STEP 4] Evaluating (run ${run})..."
  echo "  ckpt       : ${CKPT_PATH}"
  echo "  eval h5ad  : ${SCF_VALID_WITH_LATENT}"

  OUTPUT=$(python scripts/encoder_exp/scfoundation/eval_scfoundation_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${SCF_VALID_WITH_LATENT}" \
    --latent-key "X_scfoundation" \
    -n 200 \
    -o "${EVAL_OUT_PREFIX}_run_${run}.h5ad" 2>&1) || true

  echo "${OUTPUT}"
  ALL_OUTPUTS+="${OUTPUT}"$'\n'
done

# ===== Aggregate metrics to CSV =====

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
echo "###   scFoundation+DDPM pipeline for ${CELL_TYPE} complete!         ###"
echo "######################################################################"
