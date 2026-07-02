#!/bin/bash
set -e

source "scripts/lib/max_n_samples.sh"

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="scimilarity_ddpm"

TRAIN_H5="data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"

TRAIN_H5_WITH_LATENT="samples/encoder_exp/scimilarity_ddpm/task1_train_${CELL_TYPE}_with_scim_latent.h5ad"
VALID_H5_WITH_LATENT="samples/encoder_exp/scimilarity_ddpm/task1_valid_${CELL_TYPE}_with_scim_latent.h5ad"

SCIM_MODEL_DIR="/share/PertBench/checkpoints/annotation_model_v1"
LATENT_DDPM_CKPT_DIR="checkpoints/scimilarity_ddpm/latent_ddpm"

EVAL_OUT_PREFIX="samples/encoder_exp/scimilarity_ddpm/scim_latent_ddpm_mlp_task1_${CELL_TYPE}_preds"
CSV_PATH="samples/encoder_exp/scimilarity_ddpm/metrics_${CELL_TYPE}.csv"

CONFIG_PATH="configs/baselines/scimilarity_ddpm_mlp.yaml"
LOG_DIR="logs/scimilarity_ddpm"

mkdir -p \
  "samples/encoder_exp/scimilarity_ddpm" \
  "checkpoints/scimilarity_ddpm" \
  "${LATENT_DDPM_CKPT_DIR}" \
  "${LOG_DIR}"

echo "######################################################################"
echo "###   SCimilarity+DDPM pipeline for cell type: ${CELL_TYPE}"
echo "######################################################################"

ALL_OUTPUTS=""

for (( run=1; run<=NUM_RUNS; run++ )); do
  export RUN_SEED=$(($run-1))
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  echo -e "\n--- [1/4] Encode train_${CELL_TYPE} with SCimilarity (run ${run}) ---"
  python scripts/encoder_exp/scimilarity/apply_scimilarity_encoder.py \
    --data-path "${TRAIN_H5}" \
    --out-h5ad "${TRAIN_H5_WITH_LATENT}" \
    --model-dir "${SCIM_MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/encode_train_${CELL_TYPE}_run_${run}.log"

  echo -e "\n--- [2/4] Encode valid_${CELL_TYPE} with SCimilarity (run ${run}) ---"
  python scripts/encoder_exp/scimilarity/apply_scimilarity_encoder.py \
    --data-path "${VALID_H5}" \
    --out-h5ad "${VALID_H5_WITH_LATENT}" \
    --model-dir "${SCIM_MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/encode_valid_${CELL_TYPE}_run_${run}.log"

  echo -e "\n--- [3/4] Train latent DDPM+decoder for ${CELL_TYPE} (run ${run}) ---"
  python scripts/encoder_exp/scimilarity/train_scimilarity_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    --train-data-path "${TRAIN_H5_WITH_LATENT}" \
    --save-weight-dir "${LATENT_DDPM_CKPT_DIR}" 2>&1 | tee "${LOG_DIR}/train_latent_ddpm_${CELL_TYPE}_run_${run}.log"

  echo -e "\n--- [4/4] Evaluate model on valid_${CELL_TYPE} (run ${run}) ---"
  CKPT_PATH="${LATENT_DDPM_CKPT_DIR}/model_final.pth"

  OUTPUT=$(python scripts/encoder_exp/scimilarity/eval_scimilarity_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${VALID_H5_WITH_LATENT}" \
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
echo "###   SCimilarity+DDPM pipeline for ${CELL_TYPE} complete!                 ###"
echo "######################################################################"
