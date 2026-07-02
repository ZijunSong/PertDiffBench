#!/usr/bin/env bash
set -euo pipefail

IFS=$'\n\t'
trap "echo ERROR && exit 1" ERR

# =========================
# Config
# =========================
declare -A DATASETS=(
  ["ACTA2_control_coculture"]="4614"
  ["ACTA2_control_ifn"]="4559"
  ["B2M_control_coculture"]="4599"
  ["B2M_control_ifn"]="4566"
)
declare -A SAMPLE_SIZES=()
source "scripts/lib/max_n_samples.sh"
# SAMPLE_SIZES filled per dataset from test h5ad in loop

NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-scDiffusion}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# repo root ( subdirwhen to )
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "${HOMEDIR}"
echo "Current working directory: $(pwd)"

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

for dataset in "${!DATASETS[@]}"; do
  LOG_ROOT="logs/fig1/task4_1/${dataset}/scDiffusion"
  CSV_ROOT="samples/fig1/task4_1/${dataset}/scDiffusion"
  mkdir -p "${LOG_ROOT}" "${CSV_ROOT}"

  gene_size=${DATASETS[$dataset]}
  train_h5ad="data/fig1/task4/task4_${dataset}_train.h5ad"
  test_h5ad="data/fig1/task4/task4_${dataset}_test.h5ad"
  n_samples="$(max_n_samples_paired "${test_h5ad}")"

  echo "######################################################################"
  echo "###   Starting full pipeline for dataset: ${dataset}"
  echo "###   Gene Size: ${gene_size}, N Samples: ${n_samples}"
  echo "######################################################################"

  # CSV: contain mean±std + each run run originalvalue
  METRICS_CSV="${CSV_ROOT}/metrics_${dataset}.csv"

  # allevaloutput (onlyeval output)
  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    run_tag="run${i}"
    echo -e "\n=== [${dataset}] ${run_tag}: Train(VAE+Diffusion+Classifier) + Sample/Eval ==="

    # alloutput dir ( run )
    VAE_DIR="checkpoints/scdiffusion/vae_checkpoint/task4_1/${dataset}/${run_tag}"
    DIFF_DIR="checkpoints/scdiffusion/diffusion_checkpoint/task4_1/${dataset}/${run_tag}"
    CLS_DIR="checkpoints/scdiffusion/classifier_checkpoint/2-classifier/task4_1/${dataset}/${run_tag}"
    SAMP_DIR="${CSV_ROOT}/${run_tag}"
    mkdir -p "${VAE_DIR}" "${DIFF_DIR}" "${CLS_DIR}" "${SAMP_DIR}"

    # filename
    VAE_WEIGHTS="${VAE_DIR}/model_seed=0_step=9999.pt"
    DIFF_WEIGHTS="${DIFF_DIR}/my_diffusion/model010000.pt"
    CLS_WEIGHTS="${CLS_DIR}/model009999.pt"

    # file
    log_file="${LOG_ROOT}/${dataset}_${run_tag}.log"
    echo "[INFO] Log -> ${log_file}"

    {
      echo "[$(date '+%F %T')] >>> Step 1: Train VAE (${dataset}, ${run_tag})"
      ( cd src/scDiffusion/VAE && \
        python VAE_train.py \
          --data_dir "../../../${train_h5ad}" \
          --num_genes "${gene_size}" \
          --state_dict "../../../checkpoints/annotation_model_v1" \
          --save_dir "../../../${VAE_DIR}" )

      echo "[$(date '+%F %T')] >>> Step 2: Train Diffusion (${dataset}, ${run_tag})"
      ( cd src/scDiffusion && \
        python cell_train.py \
          --data_dir "../../${train_h5ad}" \
          --vae_path "../../${VAE_WEIGHTS}" \
          --save_dir "../../${DIFF_DIR}" )

      echo "[$(date '+%F %T')] >>> Step 3: Train Classifier (${dataset}, ${run_tag})"
      ( cd src/scDiffusion && \
        python classifier_train.py \
          --data_dir "../../${train_h5ad}" \
          --vae_path "../../${VAE_WEIGHTS}" \
          --model_path "../../${CLS_DIR}" )

      echo "[$(date '+%F %T')] >>> Step 4: Sampling & Evaluation (${dataset}, ${run_tag})"
    } 2>&1 | tee "${log_file}"

    # eval outputto amount ( when to )
    eval_output="$(
      ( cd src/scDiffusion && \
        python classifier_sample.py \
          --num_samples "${n_samples}" \
          --train-data-path "../../${train_h5ad}" \
          --model_path "../../${DIFF_WEIGHTS}" \
          --classifier_path "../../${CLS_WEIGHTS}" \
          --ae_dir "../../${VAE_WEIGHTS}" \
          --num_gene "${gene_size}" \
          --sample_dir "../../${SAMP_DIR}" \
          --out_h5ad "../../${SAMP_DIR}/synthetic_ifn_${i}.h5ad" \
          --umap_plot "../../${SAMP_DIR}/umap_comparison_${i}.png" \
          --init_cell_path "../../${test_h5ad}" \
        ) 2>&1 || true
    )"
    echo "${eval_output}" | tee -a "${log_file}"
    ALL_OUTPUTS+="${eval_output}\n"

    echo "[$(date '+%F %T')] >>> Finished (${dataset}, ${run_tag})" | tee -a "${log_file}"
  done

  # parseall run evaloutput, CSV ( and )
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
      if (idx==1){n=c_pds;                for(i=0;i<n;i++){v=pds[i];                s+=v}}
      else if(idx==2){n=c_mae;            for(i=0;i<n;i++){v=mae[i];                s+=v}}
      else if(idx==3){n=c_des;            for(i=0;i<n;i++){v=des[i];                s+=v}}
      else if(idx==4){n=c_edist;          for(i=0;i<n;i++){v=edist[i];              s+=v}}
      else if(idx==5){n=c_mmd;            for(i=0;i<n;i++){v=mmd[i];                s+=v}}
      else if(idx==6){n=c_r2;             for(i=0;i<n;i++){v=r2[i];                 s+=v}}
      else if(idx==7){n=c_pearson_all;    for(i=0;i<n;i++){v=pearson_all[i];        s+=v}}
      else if(idx==8){n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i]; s+=v}}
      else if(idx==9){n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i]; s+=v}}
      else if(idx==10){n=c_pearson_delta_de50; for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v}}
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

  echo -e "\n--- Finished pipeline for dataset: ${dataset} ---\n"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                          ###"
echo "######################################################################"
