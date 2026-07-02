#!/bin/bash

# Exit immediately if any command fails.
set -e

source "scripts/lib/max_n_samples.sh"

# -------------------- Config --------------------
CELL_TYPES=(
  'CD4T'
)

# noise std
NOISE_LEVELS=(0.1 0.25 0.5 1.0 1.5)

# train+eval repetitions
NUM_RUNS="${NUM_RUNS:-3}"
NUM_GENES="${NUM_GENES:-6998}" # onlyfor , andpath now
METHOD_NAME="${METHOD_NAME:-scGen-${NUM_GENES}}" # CSV name 
METHOD_DIR="${METHOD_DIR:-scgen}" # path after levelsubdirname ( name)

mkdir -p logs

# -------------------- Main loop --------------------
for cell_type in "${CELL_TYPES[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "### Processing cell type: $cell_type | noise level: $noise_level ($NUM_RUNS )"
    echo "######################################################################"

    # level suffix ( : cell_noise_noiseLevel)
    group_suffix="${cell_type}_noise_${noise_level}"

    # train data path ( lognormal noise)
    train_data_file="data/add_lognormal_bionoise_output/task1_train_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"
    # test stillusing fig1 validate (and )
    test_data_file="data/add_lognormal_bionoise_output/task1_valid_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"

    # checkfilewhetherexist, exist skip noise level
    if [ ! -f "$train_data_file" ]; then
      echo " : Training data file not found '$train_data_file'.skip this combo."
      continue
    fi

    # shared base path: only at last level use METHOD_DIR distinguish methods
    base_weight_dir="checkpoints/lognormal_bionoise/${group_suffix}/${METHOD_DIR}"
    base_samples_dir="samples/lognormal_bionoise/${group_suffix}/${METHOD_DIR}"
    mkdir -p "$base_weight_dir" "$base_samples_dir"

    all_outputs=""

    for (( i=1; i<=NUM_RUNS; i++ )); do
      export RUN_SEED=$(($i-1))
      echo -e "\n--- Running $cell_type (noise: $noise_level) run $i/$NUM_RUNS ---"

      # standaloneoutput/ directory, in directoryunder 
      model_save_dir="${base_weight_dir}/run_${i}"
      samples_dir="${base_samples_dir}/run_${i}"
      mkdir -p "$model_save_dir" "$samples_dir"

      # train + eval (eval rows )
      output=$(python scripts/scGen_eval.py \
          --train_data_path "$train_data_file" \
          --test_data_path "$test_data_file" \
          --model_save_path "$model_save_dir" \
          --out_h5ad "${samples_dir}/pred_${i}.h5ad" \
          --umap_plot "${samples_dir}/umap_comparison_${i}.png" \
          --n_samples "${N_SAMPLES}" \
          --celltype_to_predict "$cell_type" 2>&1) || true

      echo "$output"
      all_outputs+="${output}"$'\n'
    done

    # -------------------- stats and CSV output --------------------
    CSV_PATH="${base_samples_dir}/metrics_${group_suffix}.csv"
    mkdir -p "$(dirname "$CSV_PATH")"

    echo -e "\n"
    # : to mawk , can awk as gawk
    echo "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$CSV_PATH" '
      # ---------- countdefine ( in END{} ) ----------
      function print_stat(name, arr, cnt,    i,sum,mean,ssd,sd,tmp){
        if (cnt > 0){
          sum=0
          for(i=0;i<cnt;i++) sum += arr[i]+0
          mean = (cnt>0)? sum/cnt : 0
          ssd=0
          for(i=0;i<cnt;i++){ tmp = (arr[i]-mean); ssd += tmp*tmp }
          sd=(cnt>1)? sqrt(ssd/(cnt-1)) : 0
          printf "%-40s: %.4f ± %.4f\n", name, mean, sd
        } else {
          printf "%-40s: N/A (no data collected)\n", name
        }
      }

      function mean_std_str(idx,    i,sum,mean,ssd,sd,cnt,tmp){
        sum=0; ssd=0; sd=0; mean=0; cnt=0
        if(idx==1){ cnt=c_pds;                   for(i=0;i<cnt;i++) sum += pds[i]+0 }
        else if(idx==2){ cnt=c_mae;              for(i=0;i<cnt;i++) sum += mae[i]+0 }
        else if(idx==3){ cnt=c_des;              for(i=0;i<cnt;i++) sum += des[i]+0 }
        else if(idx==4){ cnt=c_edist;            for(i=0;i<cnt;i++) sum += edist[i]+0 }
        else if(idx==5){ cnt=c_mmd;              for(i=0;i<cnt;i++) sum += mmd[i]+0 }
        else if(idx==6){ cnt=c_r2;               for(i=0;i<cnt;i++) sum += r2[i]+0 }
        else if(idx==7){ cnt=c_pearson_all;      for(i=0;i<cnt;i++) sum += pearson_all[i]+0 }
        else if(idx==8){ cnt=c_pearson_delta_all;for(i=0;i<cnt;i++) sum += pearson_delta_all[i]+0 }
        else if(idx==9){ cnt=c_pearson_delta_de20;for(i=0;i<cnt;i++) sum += pearson_delta_de20[i]+0 }
        else if(idx==10){cnt=c_pearson_delta_de50;for(i=0;i<cnt;i++) sum += pearson_delta_de50[i]+0 }
        else if(idx==11){cnt=c_pearson_delta_de100;for(i=0;i<cnt;i++) sum += pearson_delta_de100[i]+0 }

        if(cnt>0){
          mean = sum/cnt
          if(idx==1){ for(i=0;i<cnt;i++){ tmp=pds[i]-mean; ssd+=tmp*tmp } }
          else if(idx==2){ for(i=0;i<cnt;i++){ tmp=mae[i]-mean; ssd+=tmp*tmp } }
          else if(idx==3){ for(i=0;i<cnt;i++){ tmp=des[i]-mean; ssd+=tmp*tmp } }
          else if(idx==4){ for(i=0;i<cnt;i++){ tmp=edist[i]-mean; ssd+=tmp*tmp } }
          else if(idx==5){ for(i=0;i<cnt;i++){ tmp=mmd[i]-mean; ssd+=tmp*tmp } }
          else if(idx==6){ for(i=0;i<cnt;i++){ tmp=r2[i]-mean; ssd+=tmp*tmp } }
          else if(idx==7){ for(i=0;i<cnt;i++){ tmp=pearson_all[i]-mean; ssd+=tmp*tmp } }
          else if(idx==8){ for(i=0;i<cnt;i++){ tmp=pearson_delta_all[i]-mean; ssd+=tmp*tmp } }
          else if(idx==9){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de20[i]-mean; ssd+=tmp*tmp } }
          else if(idx==10){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de50[i]-mean; ssd+=tmp*tmp } }
          else if(idx==11){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de100[i]-mean; ssd+=tmp*tmp } }
          sd = (cnt>1)? sqrt(ssd/(cnt-1)) : 0
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

      function mean_std_str_wrap(i,   ms,parts){
        ms = mean_std_str(i); split(ms,parts,"|")
        return sprintf("%.4f±%.4f", parts[1]+0, parts[2]+0)
      }

      # ---------- collect metrics ----------
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF+0 }
      /Mean Absolute Error \(MAE\):/             { mae[c_mae++] = $NF+0 }
      /Differential Expression Score \(DES\):/   { des[c_des++] = $NF+0 }
      /E-Distance:/                              { edist[c_edist++] = $NF+0 }
      /Maximum Mean Discrepancy \(MMD\):/        { mmd[c_mmd++] = $NF+0 }
      /R-squared \(R2\):/                        { r2[c_r2++] = $NF+0 }
      /Pearson \(all genes\):/                   { pearson_all[c_pearson_all++] = $NF+0 }
      /Pearson Delta \(all genes\):/             { pearson_delta_all[c_pearson_delta_all++] = $NF+0 }
      /Pearson Delta \(top 20 DE genes\):/       { pearson_delta_de20[c_pearson_delta_de20++] = $NF+0 }
      /Pearson Delta \(top 50 DE genes\):/       { pearson_delta_de50[c_pearson_delta_de50++] = $NF+0 }
      /Pearson Delta \(top 100 DE genes\):/      { pearson_delta_de100[c_pearson_delta_de100++] = $NF+0 }

      END{
        print "=================================================================="
        printf " %s (noise: %s) final stats (%d )\n", dataset, noise, num_runs
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

        # -------- CSV (Dataset, Noise, Method + mean±std and originalvalue)--------
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

        header = "Dataset,Noise,Method"
        for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)"
        for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i]

        row = dataset "," noise "," method
        for (i=1;i<=11;i++) row = row "," mean_std_str_wrap(i)
        for (r=0;r<num_runs;r++){
          for (i=1;i<=11;i++){
            v = val_idx(i, r)
            if (v == "") row = row ","
            else row = row sprintf(",%.4f", v+0)
          }
        }

        print header > csv_path
        print row    >> csv_path
        close(csv_path)
        printf("CSV written: %s\n", csv_path)
      }
    '

    echo -e "\n--- Done: $cell_type | noise: $noise_level ---\n"
  done
done

echo "######################################################################"
echo "###   All cell types and noise levels finished!                 ###"
echo "######################################################################"
