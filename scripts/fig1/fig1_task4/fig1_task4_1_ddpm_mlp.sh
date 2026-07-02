#!/usr/bin/env bash
set -euo pipefail

IFS=$'\n\t'

# ==================== Config ====================
declare -A GENE_SIZES=(
  ["ACTA2_control_coculture"]="4614"
  ["ACTA2_control_ifn"]="4559"
  ["B2M_control_coculture"]="4599"
  ["B2M_control_ifn"]="4566"
)
declare -A SAMPLE_SIZES=()
source "scripts/lib/max_n_samples.sh"
# SAMPLE_SIZES filled per dataset from test h5ad in loop

NUM_RUNS=${NUM_RUNS:-3}
CONFIG_FILE="${CONFIG_FILE:-configs/baselines/mlp_ddpm_mlp.yaml}"
METHOD_NAME="${METHOD_NAME:-MLP-DDPM-MLP}"

# eval mustin under 11 ( ascountvalue): 
# Perturbation Discrimination Score (PDS): <num>
# Mean Absolute Error (MAE): <num>
# Differential Expression Score (DES): <num>
# E-Distance: <num>
# Maximum Mean Discrepancy (MMD): <num>
# R-squared (R2): <num>
# Pearson (all genes): <num>
# Pearson Delta (all genes): <num>
# Pearson Delta (top 20 DE genes): <num>
# Pearson Delta (top 50 DE genes): <num>
# Pearson Delta (top 100 DE genes): <num>

for dataset in "${!GENE_SIZES[@]}"; do
  # path mlp_ddpm_mlp 
  LOG_ROOT="logs/fig1/task4_1/${dataset}/mlp_ddpm_mlp"
  CSV_ROOT="samples/fig1/task4_1/${dataset}/mlp_ddpm_mlp"
  mkdir -p "$LOG_ROOT" "$CSV_ROOT"

  gene_size=${GENE_SIZES[$dataset]}
  train_data_path="data/fig1/task4/task4_${dataset}_train.h5ad"
  valid_data_path="data/fig1/task4/task4_${dataset}_test.h5ad"
  n_samples="$(max_n_samples_paired "${valid_data_path}")"

  echo "######################################################################"
  echo "###   Starting pipeline for dataset: $dataset"
  echo "###   Gene Size: $gene_size, N Samples: $n_samples"
  echo "######################################################################"


  # CSV: contain mean±std + each run run originalvalue
  METRICS_CSV="${CSV_ROOT}/metrics_${dataset}.csv"

  # allevaloutput (onlyeval output)
  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo -e "\n=== [${dataset}] Run ${i}/${NUM_RUNS}: Train + Eval ==="
    run_tag="run${i}"

    ckpt_dir="checkpoints/ddpm_mlp/${dataset}/${run_tag}"
    samples_dir="samples/fig1/task4_1/${dataset}/mlp_ddpm_mlp/${run_tag}"
    mkdir -p "$ckpt_dir" "$samples_dir"

    log_file="${LOG_ROOT}/${dataset}_${run_tag}.log"
    echo "[INFO] Logs -> ${log_file}"

    {
      echo "[$(date '+%F %T')] >>> Step 1: Training (${dataset}, ${run_tag})"
      python scripts/baseline/train_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "$train_data_path" \
        --save-weight-dir "$ckpt_dir" \
        --gene-nums "$gene_size"
    } 2>&1 | tee "$log_file"

    echo "[$(date '+%F %T')] >>> Step 2: Evaluation (${dataset}, ${run_tag})" | tee -a "$log_file"
    # eval output; output 
    eval_output="$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "$valid_data_path" \
        --ckpt "${ckpt_dir}/model_epoch_1000.pth" \
        --out_h5ad "${samples_dir}/synthetic_ifn_${i}.h5ad" \
        --gene-nums "$gene_size" \
        --umap_plot "${samples_dir}/umap_comparison_${i}.png" \
        --train-data-path "$train_data_path" \
        --n_samples "$n_samples" 2>&1 || true)"
    echo "${eval_output}" | tee -a "$log_file"
    ALL_OUTPUTS+="${eval_output}\n"

    echo "[$(date '+%F %T')] >>> Finished (${dataset}, ${run_tag})" | tee -a "$log_file"
  done

  # parseall run evaloutput, CSV
  echo -e "${ALL_OUTPUTS}" | awk -v ds="${dataset}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${METRICS_CSV}" '
    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
    /Mean Absolute Error \(MAE\):/               { mae[c_mae++] = $NF }
    /Differential Expression Score \(DES\):/     { des[c_des++] = $NF }
    /^E-Distance:/                               { edist[c_edist++] = $NF }
    /Maximum Mean Discrepancy \(MMD\):/          { mmd[c_mmd++] = $NF }
    /R-squared \(R2\):/                          { r2[c_r2++] = $NF }
    /Pearson \(all genes\):/                     { pearson_all[c_pearson_all++] = $NF }
    /Pearson Delta \(all genes\):/               { pearson_delta_all[c_pearson_delta_all++] = $NF }
    /Pearson Delta \(top 20 DE genes\):/         { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
    /Pearson Delta \(top 50 DE genes\):/         { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
    /Pearson Delta \(top 100 DE genes\):/        { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

    function mean_std(idx,    i,n,s,mu,ss,v) {
      if (idx==1){n=c_pds;   for(i=0;i<n;i++){v=pds[i];s+=v}}
      else if(idx==2){n=c_mae;for(i=0;i<n;i++){v=mae[i];s+=v}}
      else if(idx==3){n=c_des;for(i=0;i<n;i++){v=des[i];s+=v}}
      else if(idx==4){n=c_edist;for(i=0;i<n;i++){v=edist[i];s+=v}}
      else if(idx==5){n=c_mmd; for(i=0;i<n;i++){v=mmd[i]; s+=v}}
      else if(idx==6){n=c_r2;  for(i=0;i<n;i++){v=r2[i];  s+=v}}
      else if(idx==7){n=c_pearson_all;for(i=0;i<n;i++){v=pearson_all[i];s+=v}}
      else if(idx==8){n=c_pearson_delta_all;for(i=0;i<n;i++){v=pearson_delta_all[i];s+=v}}
      else if(idx==9){n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];s+=v}}
      else if(idx==10){n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i];s+=v}}
      else if(idx==11){n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i];s+=v}}
      mu = (n>0)? s/n : 0;
      for(i=0;i<n;i++){
        if (idx==1) v=pds[i];
        else if(idx==2) v=mae[i];
        else if(idx==3) v=des[i];
        else if(idx==4) v=edist[i];
        else if(idx==5) v=mmd[i];
        else if(idx==6) v=r2[i];
        else if(idx==7) v=pearson_all[i];
        else if(idx==8) v=pearson_delta_all[i];
        else if(idx==9) v=pearson_delta_de20[i];
        else if(idx==10) v=pearson_delta_de50[i];
        else if(idx==11) v=pearson_delta_de100[i];
        ss += (v - mu) * (v - mu);
      }
      return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
    }

    function val(idx, r,    v){
      if (idx==1) v=pds[r];
      else if(idx==2) v=mae[r];
      else if(idx==3) v=des[r];
      else if(idx==4) v=edist[r];
      else if(idx==5) v=mmd[r];
      else if(idx==6) v=r2[r];
      else if(idx==7) v=pearson_all[r];
      else if(idx==8) v=pearson_delta_all[r];
      else if(idx==9) v=pearson_delta_de20[r];
      else if(idx==10) v=pearson_delta_de50[r];
      else if(idx==11) v=pearson_delta_de100[r];
      return (v=="") ? 0 : v;
    }

    END {
      metric_names[1]="PDS";
      metric_names[2]="MAE";
      metric_names[3]="DES";
      metric_names[4]="E-Distance";
      metric_names[5]="MMD";
      metric_names[6]="R2";
      metric_names[7]="Pearson (all genes)";
      metric_names[8]="Pearson Delta (all genes)";
      metric_names[9]="Pearson Delta (top 20 DE genes)";
      metric_names[10]="Pearson Delta (top 50 DE genes)";
      metric_names[11]="Pearson Delta (top 100 DE genes)";

      header = "Dataset,Method";
      for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
      for (r=1;r<=num_runs;r++)
        for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];

      row = ds "," method;
      for (i=1;i<=11;i++) {
        ms = mean_std(i); split(ms, parts, "|");
        row = row sprintf(",%.6f±%.6f", parts[1], parts[2]);
      }
      for (r=0;r<num_runs;r++) {
        for (i=1;i<=11;i++) row = row sprintf(",%.6f", val(i, r));
      }

      print header > csv_path; # eachdata ownheader
      print row >> csv_path; # and data 
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished pipeline for dataset: $dataset ---\n"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                          ###"
echo "######################################################################"
