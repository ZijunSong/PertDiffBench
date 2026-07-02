#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ========= Configuration =========
# Path prefix; convention: data under data/highly_variable_gene_gradient/; checkpoints under CKPT_ROOT/fig1/task1/<method>/<cell_type>_hvg_1000; samples under samples/fig1/task1/<cell_type>/<method>_1000; logs under logs/fig1_task1
ROOT_DIR="${ROOT_DIR:-/data/ppnm/data/PertDiffBench/}"
CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"
NUM_RUNS=3
METHOD_NAME="Squidiff"
LOG_ROOT="${ROOT_DIR}logs/fig1_task1"
mkdir -p "${LOG_ROOT}"

# n_samples per cell type (max paired cells in valid set)
source "scripts/lib/max_n_samples.sh"
declare -A SAMPLES_MAP=()
build_samples_map_from_valid_h5ad "${ROOT_DIR}" "${CELL_TYPES[@]}"

CELL_TYPES=(
  'B'
  'CD4T'
  'CD8T'
  'CD14+Mono'
  'Dendritic'
  'FCGR3A+Mono'
  'NK'
)

# Loop through each cell type
for cell_type in "${CELL_TYPES[@]}"; do
  echo "######################################################################"
  echo "###   Starting to process cell type: $cell_type"
  echo "######################################################################"

  n_samples=${SAMPLES_MAP[$cell_type]}
  [ -z "$n_samples" ] && { echo "No n_samples configured for ${cell_type}"; exit 1; }

  # Paths (same convention across fig1 task1 scripts)
  save_dir_base="${CKPT_ROOT}/fig1/task1/squidiff/${cell_type}_hvg_1000"
  sample_dir_base="${ROOT_DIR}samples/fig1/task1/${cell_type}/squidiff_1000"
  train_path="${ROOT_DIR}data/highly_variable_gene_gradient/${cell_type}_train_HVG_1000.h5ad"
  valid_path="${ROOT_DIR}data/highly_variable_gene_gradient/${cell_type}_valid_HVG_1000.h5ad"
  mkdir -p "${save_dir_base}" "${sample_dir_base}"
  csv_path="${sample_dir_base}/metrics_${METHOD_NAME}_${cell_type}_hvg_1000.csv"

  # Step 1: Train the model for the current cell type (once per cell type)
  echo -e "\n--- Training model for $cell_type ---"
  python src/Squidiff/train_squidiff.py \
    --logger_path "${LOG_ROOT}" \
    --data_path "${train_path}" \
    --resume_checkpoint "${save_dir_base}" \
    --gene_size 1000 \
    --output_dim 1000 2>&1 | tee "${LOG_ROOT}/train_${cell_type}.log"

  echo "--- Training for $cell_type complete. ---"

  all_outputs=""

  # Step 2: Run inference NUM_RUNS times for the current cell type
  echo -e "\n--- Starting inference for $cell_type ($NUM_RUNS runs total) ---"
  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo -e "\n--- Running inference iteration $i/$NUM_RUNS for $cell_type ---"
    sample_dir_run="${sample_dir_base}/run${i}"
    mkdir -p "${sample_dir_run}"
    output=$(python src/Squidiff/sample_squidiff.py \
      --model_path "${save_dir_base}/model.pt" \
      --gene_size 1000 \
      --output_dim 1000 \
      --out_h5ad "${sample_dir_run}/synthetic_ifn_run_${i}.h5ad" \
      --train_data_path "${valid_path}" \
      --n_samples "${n_samples}" \
      --umap_plot "${sample_dir_run}/umap_comparison_${i}.png" \
      --data_path "${valid_path}" 2>&1) || true

    echo "$output"
    all_outputs+="$output\n"
  done

    # Step 3: Aggregate stats and write CSV
    echo -e "\n"
    echo -e "$all_outputs" | awk -v dataset="$cell_type" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$csv_path" '
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = $NF }
      /Differential Expression Score \(DES\):/    { des[c_des++] = $NF }
      /E-Distance:/                               { edist[c_edist++] = $NF }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = $NF }
      /R-squared \(R2\):/                         { r2[c_r2++] = $NF }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++] = $NF }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = $NF }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

      function mean_std(idx,    i,n,s,mu,ss,v) {
        if (idx==1)  { n=c_pds;   for(i=0;i<n;i++){ v=pds[i];   s+=v } }
        else if(idx==2){ n=c_mae;   for(i=0;i<n;i++){ v=mae[i];   s+=v } }
        else if(idx==3){ n=c_des;   for(i=0;i<n;i++){ v=des[i];   s+=v } }
        else if(idx==4){ n=c_edist; for(i=0;i<n;i++){ v=edist[i]; s+=v } }
        else if(idx==5){ n=c_mmd;   for(i=0;i<n;i++){ v=mmd[i];   s+=v } }
        else if(idx==6){ n=c_r2;    for(i=0;i<n;i++){ v=r2[i];    s+=v } }
        else if(idx==7){ n=c_pearson_all;      for(i=0;i<n;i++){ v=pearson_all[i];      s+=v } }
        else if(idx==8){ n=c_pearson_delta_all;for(i=0;i<n;i++){ v=pearson_delta_all[i];s+=v } }
        else if(idx==9){ n=c_pearson_delta_de20; for(i=0;i<n;i++){ v=pearson_delta_de20[i]; s+=v } }
        else if(idx==10){ n=c_pearson_delta_de50; for(i=0;i<n;i++){ v=pearson_delta_de50[i]; s+=v } }
        else if(idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){ v=pearson_delta_de100[i];s+=v } }
        mu = (n>0) ? s/n : 0;
        ss = 0;
        for(i=0;i<n;i++) {
          if(idx==1) v=pds[i]; else if(idx==2) v=mae[i]; else if(idx==3) v=des[i];
          else if(idx==4) v=edist[i]; else if(idx==5) v=mmd[i]; else if(idx==6) v=r2[i];
          else if(idx==7) v=pearson_all[i]; else if(idx==8) v=pearson_delta_all[i];
          else if(idx==9) v=pearson_delta_de20[i]; else if(idx==10) v=pearson_delta_de50[i];
          else if(idx==11) v=pearson_delta_de100[i];
          ss += (v - mu)^2;
        }
        return (n>1) ? mu "|" sqrt(ss/(n-1)) : mu "|0";
      }
      function val(idx, j,    v) {
        if (idx==1) v=pds[j]; else if(idx==2) v=mae[j]; else if(idx==3) v=des[j];
        else if(idx==4) v=edist[j]; else if(idx==5) v=mmd[j]; else if(idx==6) v=r2[j];
        else if(idx==7) v=pearson_all[j]; else if(idx==8) v=pearson_delta_all[j];
        else if(idx==9) v=pearson_delta_de20[j]; else if(idx==10) v=pearson_delta_de50[j];
        else if(idx==11) v=pearson_delta_de100[j]; return v+0;
      }
      function print_stat(name, data, count,    i,sum,mu,ss,std) {
        if (count > 0) {
          sum = 0; for (i = 0; i < count; i++) sum += data[i];
          mu = sum / count;
          ss = 0; for (i = 0; i < count; i++) ss += (data[i] - mu)^2;
          std = (count > 1) ? sqrt(ss / (count - 1)) : 0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std;
        } else { printf "%-40s: N/A (No data)\n", name; }
      }

      END {
        print "==================================================================";
        printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
        print "==================================================================";
        print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
        print_stat("Mean Absolute Error (MAE)", mae, c_mae);
        print_stat("Differential Expression Score (DES)", des, c_des);
        print "----------------------------------------";
        print_stat("E-Distance", edist, c_edist);
        print_stat("Maximum Mean Discrepancy (MMD)", mmd, c_mmd);
        print_stat("R-squared (R2)", r2, c_r2);
        print "----------------------------------------";
        print_stat("Pearson (all genes)", pearson_all, c_pearson_all);
        print_stat("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all);
        print_stat("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20);
        print_stat("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50);
        print_stat("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100);
        print "==================================================================\n";

        metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES";
        metric_names[4]="E-Distance"; metric_names[5]="MMD"; metric_names[6]="R2";
        metric_names[7]="Pearson (all genes)"; metric_names[8]="Pearson Delta (all genes)";
        metric_names[9]="Pearson Delta (top 20 DE genes)"; metric_names[10]="Pearson Delta (top 50 DE genes)";
        metric_names[11]="Pearson Delta (top 100 DE genes)";
        header = "Method";
        for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
        for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];
        row = method;
        for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
        for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));
        print header > csv_path;
        print row   >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    '

    echo -e "\n--- Finished pipeline for cell type: $cell_type ---\n"
done

echo "######################################################################"
echo "###   All cell type processing is complete!                        ###"
echo "######################################################################"
