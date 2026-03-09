#!/bin/bash
# Exit on first error
set -e

# Consistent numeric locale for awk
export LC_ALL=C LC_NUMERIC=C

# ==============================================================================
# Script Configuration
# ==============================================================================
CELL_TYPES=('CD4T')

NOISE_LEVELS=(
  '0.25'
  '0.5'
  '1.0'
  '2.0'
  '4.0'
)

NUM_GENES="6998"
# Now NUM_RUNS means how many times to repeat the whole train+eval pipeline
NUM_RUNS=3

CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"
BASE_DATA_DIR="data/add_poisson_technoise_output"
METHOD_NAME="scRNA-DDPM-scRNA-${NUM_GENES}"

mkdir -p "logs"

# ==============================================================================
# Main Processing Loop
# ==============================================================================
for cell_type in "${CELL_TYPES[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###   Pipeline starting for: Cell Type = $cell_type | Noise Std = $noise_level"
    echo "######################################################################"

    # --- Dynamic paths (data) ---
    train_data_path="${BASE_DATA_DIR}/task1_train_${cell_type}_exp_poisson_depth_${noise_level}.h5ad"
    valid_data_path="${BASE_DATA_DIR}/task1_valid_${cell_type}_exp_poisson_depth_${noise_level}.h5ad"

    # --- Base output dirs (method-level) ---
    base_weight_dir="checkpoints/poisson_technoise/${cell_type}_noise_${noise_level}/mlp_ddpm_mlp"
    base_samples_dir="samples/poisson_technoise/${cell_type}_noise_${noise_level}/mlp_ddpm_mlp"
    mkdir -p "$base_weight_dir" "$base_samples_dir"

    # Will collect stdout/stderr from each run to aggregate metrics
    all_outputs=""

    echo -e "\n--- Repeating Train+Eval for $cell_type with noise $noise_level: ${NUM_RUNS} runs ---"

    for (( run_id=1; run_id<=NUM_RUNS; run_id++ )); do
      echo -e "\n=== Run ${run_id}/${NUM_RUNS} | ${cell_type}, Noise: ${noise_level} ==="

      # --- Per-run directories ---
      save_weight_dir="${base_weight_dir}/run_${run_id}"
      samples_dir="${base_samples_dir}/run_${run_id}"
      mkdir -p "$save_weight_dir" "$samples_dir"

      checkpoint_file="${save_weight_dir}/model_epoch_1000.pth"

      # --- Step 1: Training (per run) ---
      echo -e "\n--- [Run ${run_id}] Training model ---"
      python scripts/baseline/train_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "$train_data_path" \
        --save-weight-dir "$save_weight_dir" \
        --gene-nums "$NUM_GENES" 2>&1 | tee "logs/train_${cell_type}_noise_${noise_level}_run${run_id}.log"

      echo -e "\n--- [Run ${run_id}] Training complete. Checkpoint: ${checkpoint_file} ---"

      # --- Step 2: Single evaluation for this run ---
      echo -e "\n--- [Run ${run_id}] Evaluating ---"
      # capture stdout+stderr; do not stop the whole pipeline on python eval failure
      output=$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "$valid_data_path" \
        --train-data-path "$train_data_path" \
        --ckpt "$checkpoint_file" \
        --out_h5ad "${samples_dir}/synthetic_ifn_run_${run_id}.h5ad" \
        --gene-nums "$NUM_GENES" \
        --umap_plot "${samples_dir}/umap_comparison_run_${run_id}.png" \
        --n_samples 6 2>&1) || true

      echo "$output" | tee "logs/eval_${cell_type}_noise_${noise_level}_run${run_id}.log"
      # Preserve real newlines into the collector
      all_outputs+="${output}"$'\n'

      echo -e "\n--- [Run ${run_id}] Done. ---"
    done

    # --- Step 3: Stats + CSV across the 3 runs (each run = one eval) ---
    echo -e "\n--- Step 3: Calculating statistics & writing CSV (across ${NUM_RUNS} runs) ---"

    # One CSV per (cell_type, noise_level) group
    CSV_PATH="${base_samples_dir}/metrics_${cell_type}_noise_${noise_level}.csv"
    mkdir -p "$(dirname "$CSV_PATH")"

    echo "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" \
                         -v method="$METHOD_NAME" -v csv_path="$CSV_PATH" '
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

      function print_stat(name, arr, cnt,   i,sum,mean,ssd,sd){
        if (cnt > 0){
          sum=0; for(i=0;i<cnt;i++) sum+=arr[i]
          mean=sum/cnt; ssd=0; for(i=0;i<cnt;i++) ssd+=(arr[i]-mean)^2
          sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
          printf "%-40s: %.4f ± %.4f\n", name, mean, sd
        } else {
          printf "%-40s: N/A (No data collected)\n", name
        }
      }

      function mean_std_str(idx,   i,sum,mean,ssd,sd,cnt){
        if     (idx==1){ cnt=c_pds;               for(i=0;i<cnt;i++) sum+=pds[i] }
        else if(idx==2){ cnt=c_mae;               for(i=0;i<cnt;i++) sum+=mae[i] }
        else if(idx==3){ cnt=c_des;               for(i=0;i<cnt;i++) sum+=des[i] }
        else if(idx==4){ cnt=c_edist;             for(i=0;i<cnt;i++) sum+=edist[i] }
        else if(idx==5){ cnt=c_mmd;               for(i=0;i<cnt;i++) sum+=mmd[i] }
        else if(idx==6){ cnt=c_r2;                for(i=0;i<cnt;i++) sum+=r2[i] }
        else if(idx==7){ cnt=c_pearson_all;       for(i=0;i<cnt;i++) sum+=pearson_all[i] }
        else if(idx==8){ cnt=c_pearson_delta_all; for(i=0;i<cnt;i++) sum+=pearson_delta_all[i] }
        else if(idx==9){ cnt=c_pearson_delta_de20;for(i=0;i<cnt;i++) sum+=pearson_delta_de20[i] }
        else if(idx==10){cnt=c_pearson_delta_de50;for(i=0;i<cnt;i++) sum+=pearson_delta_de50[i] }
        else if(idx==11){cnt=c_pearson_delta_de100;for(i=0;i<cnt;i++) sum+=pearson_delta_de100[i] }

        if(cnt>0){
          mean=sum/cnt; ssd=0
          if     (idx==1){ for(i=0;i<cnt;i++) ssd+=(pds[i]-mean)^2 }
          else if(idx==2){ for(i=0;i<cnt;i++) ssd+=(mae[i]-mean)^2 }
          else if(idx==3){ for(i=0;i<cnt;i++) ssd+=(des[i]-mean)^2 }
          else if(idx==4){ for(i=0;i<cnt;i++) ssd+=(edist[i]-mean)^2 }
          else if(idx==5){ for(i=0;i<cnt;i++) ssd+=(mmd[i]-mean)^2 }
          else if(idx==6){ for(i=0;i<cnt;i++) ssd+=(r2[i]-mean)^2 }
          else if(idx==7){ for(i=0;i<cnt;i++) ssd+=(pearson_all[i]-mean)^2 }
          else if(idx==8){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_all[i]-mean)^2 }
          else if(idx==9){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de20[i]-mean)^2 }
          else if(idx==10){for(i=0;i<cnt;i++) ssd+=(pearson_delta_de50[i]-mean)^2 }
          else if(idx==11){for(i=0;i<cnt;i++) ssd+=(pearson_delta_de100[i]-mean)^2 }
          sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
          return sprintf("%.4f|%.4f", mean, sd)
        }
        return "0.0000|0.0000"
      }

      function val_idx(idx, r,   v){
        if     (idx==1){  v = (r < c_pds)?pds[r]:"" }
        else if(idx==2){  v = (r < c_mae)?mae[r]:"" }
        else if(idx==3){  v = (r < c_des)?des[r]:"" }
        else if(idx==4){  v = (r < c_edist)?edist[r]:"" }
        else if(idx==5){  v = (r < c_mmd)?mmd[r]:"" }
        else if(idx==6){  v = (r < c_r2)?r2[r]:"" }
        else if(idx==7){  v = (r < c_pearson_all)?pearson_all[r]:"" }
        else if(idx==8){  v = (r < c_pearson_delta_all)?pearson_delta_all[r]:"" }
        else if(idx==9){  v = (r < c_pearson_delta_de20)?pearson_delta_de20[r]:"" }
        else if(idx==10){ v = (r < c_pearson_delta_de50)?pearson_delta_de50[r]:"" }
        else if(idx==11){ v = (r < c_pearson_delta_de100)?pearson_delta_de100[r]:"" }
        return v
      }

      END{
        print "=================================================================="
        printf " Final statistics for %s (Noise Std: %s) (%d runs)\n", dataset, noise, num_runs
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

        # CSV header with mean±std and per-run values
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

        # write CSV (overwrite for each (cell,noise) group)
        print header > csv_path
        print row    >> csv_path
        close(csv_path)
        printf("CSV written: %s\n", csv_path)
      }
    '

    echo -e "\n--- Finished pipeline for: Cell Type = $cell_type | Noise Std = $noise_level ---\n"
  done
done

echo "######################################################################"
echo "###   All processing is complete!                                  ###"
echo "######################################################################"
