#!/usr/bin/env bash

# Exit on error; treat unset vars as error; fail on pipe errors
set -euo pipefail

# ======================== Config ========================
# Gene counts to process (edit as needed)
GENE_NUMS_LIST=(6998 6000 5000 4000 3000 2000 1000)

# Number of RUNS = (train + eval) times
NUM_RUNS=3

# Method name for CSV first column
METHOD_NAME="Squidiff"

# Paths
CELL_TYPE="CD4T"
CHECKPOINT_ROOT="checkpoints/squidiff"
SAMPLE_ROOT="samples/highly_variable_gene_gradient"
LOG_ROOT="logs/squidiff"

mkdir -p "${CHECKPOINT_ROOT}" "${SAMPLE_ROOT}" "${LOG_ROOT}"

# ====================== Main Loop =======================
for gene_num in "${GENE_NUMS_LIST[@]}"; do
  echo "######################################################################"
  echo "###   Starting Squidiff pipeline for: ${CELL_TYPE}, Genes = ${gene_num}"
  echo "######################################################################"

  # ---- Data paths (fixed for this gene_num) ----
  train_data_path="data/highly_variable_gene_gradient/${CELL_TYPE}_train_HVG_${gene_num}.h5ad"
  valid_data_path="data/highly_variable_gene_gradient/${CELL_TYPE}_valid_HVG_${gene_num}.h5ad"

  # A combined log per gene_num
  ts_all=$(date +%Y%m%d_%H%M%S)
  gene_log_file="${LOG_ROOT}/squidiff_${CELL_TYPE}_hvg_${gene_num}_ALL_${ts_all}.log"

  # Base dir for samples (per gene)
  sample_dir_base="${SAMPLE_ROOT}/squidiff_${gene_num}"
  mkdir -p "${sample_dir_base}"

  # We will collect all evaluation outputs across runs in this variable
  all_outputs=""

  {
    echo
    echo "--- 3 Runs: (Train + Eval) for Genes=${gene_num} ---"

    for (( run=1; run<=NUM_RUNS; run++ )); do
      echo
      echo "====================== RUN ${run}/${NUM_RUNS} ======================"

      # ---- Per-run paths ----
      checkpoint_dir="${CHECKPOINT_ROOT}/${CELL_TYPE}_hvg_${gene_num}/run${run}"
      model_file="${checkpoint_dir}/model.pt"
      run_dir="${sample_dir_base}/run${run}"
      mkdir -p "${checkpoint_dir}" "${run_dir}"

      ts=$(date +%Y%m%d_%H%M%S)
      run_log="${LOG_ROOT}/squidiff_${CELL_TYPE}_hvg_${gene_num}_run${run}_${ts}.log"

      echo "--- Step 1: Training (run ${run}) ---"
      # 如果你的训练脚本支持 --seed，可追加： --seed "${run}"
      python src/Squidiff/train_squidiff.py \
        --logger_path "${LOG_ROOT}" \
        --data_path "${train_data_path}" \
        --resume_checkpoint "${checkpoint_dir}" \
        --gene_size "${gene_num}" \
        --output_dim "${gene_num}"

      echo
      echo "--- Step 2: Sampling + Evaluation (run ${run}) ---"
      # 若采样脚本支持保存 h5ad，这里会写到 run 目录
      # 若不支持，请去掉 --out_h5ad 这个参数
      output=$(python src/Squidiff/sample_squidiff.py \
        --model_path "${model_file}" \
        --train_data_path "${train_data_path}" \
        --gene_size "${gene_num}" \
        --output_dim "${gene_num}" \
        --n_samples 278 \
        --data_path "${valid_data_path}" \
        --out_h5ad "${run_dir}/synthetic_ifn_run${run}.h5ad" 2>&1) || true

      echo "$output" | tee -a "${run_log}"
      all_outputs+=$'\n'"$output"
    done

    echo
    echo "--- Step 3: Aggregate stats across ${NUM_RUNS} runs + Write CSV ---"
    csv_file="${sample_dir_base}/metrics_squidiff_gene_${gene_num}.csv"

    # Force C locale to ensure decimal parsing in awk
    LC_ALL=C \
    printf '%s\n' "$all_outputs" | awk -v cell_type="${CELL_TYPE}" -v gene_count="${gene_num}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_file}" '
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF+0 }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = $NF+0 }
      /Differential Expression Score \(DES\):/    { des[c_des++] = $NF+0 }
      /E-Distance:/                               { edist[c_edist++] = $NF+0 }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = $NF+0 }
      /R-squared \(R2\):/                         { r2[c_r2++] = $NF+0 }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++] = $NF+0 }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = $NF+0 }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = $NF+0 }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = $NF+0 }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = $NF+0 }

      function print_stat(name, data, count,    i,sum,mu,ss,std){
        if(count>0){
          sum=0; for(i=0;i<count;i++) sum+=data[i];
          mu=sum/count; ss=0; for(i=0;i<count;i++) ss+=(data[i]-mu)^2;
          std=(count>1)? sqrt(ss/(count-1)) : 0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std;
        } else {
          printf "%-40s: N/A (No data)\n", name;
        }
      }

      # for CSV: get mean|std string for metric idx
      function mean_std(idx,    i,n,s,mu,ss,v){
        if(idx==1){n=c_pds; for(i=0;i<n;i++){v=pds[i]; s+=v}}
        else if(idx==2){n=c_mae; for(i=0;i<n;i++){v=mae[i]; s+=v}}
        else if(idx==3){n=c_des; for(i=0;i<n;i++){v=des[i]; s+=v}}
        else if(idx==4){n=c_edist; for(i=0;i<n;i++){v=edist[i]; s+=v}}
        else if(idx==5){n=c_mmd; for(i=0;i<n;i++){v=mmd[i]; s+=v}}
        else if(idx==6){n=c_r2; for(i=0;i<n;i++){v=r2[i]; s+=v}}
        else if(idx==7){n=c_pearson_all; for(i=0;i<n;i++){v=pearson_all[i]; s+=v}}
        else if(idx==8){n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i]; s+=v}}
        else if(idx==9){n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i]; s+=v}}
        else if(idx==10){n=c_pearson_delta_de50; for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v}}
        else if(idx==11){n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v}}
        mu=(n>0)? s/n:0; ss=0; for(i=0;i<n;i++){
          if(idx==1) v=pds[i]; else if(idx==2) v=mae[i]; else if(idx==3) v=des[i];
          else if(idx==4) v=edist[i]; else if(idx==5) v=mmd[i]; else if(idx==6) v=r2[i];
          else if(idx==7) v=pearson_all[i]; else if(idx==8) v=pearson_delta_all[i];
          else if(idx==9) v=pearson_delta_de20[i]; else if(idx==10) v=pearson_delta_de50[i];
          else if(idx==11) v=pearson_delta_de100[i]; ss+=(v-mu)*(v-mu);
        }
        return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
      }
      # per-run value accessor
      function val(idx,j, v){
        if(idx==1) v=pds[j]; else if(idx==2) v=mae[j]; else if(idx==3) v=des[j];
        else if(idx==4) v=edist[j]; else if(idx==5) v=mmd[j]; else if(idx==6) v=r2[j];
        else if(idx==7) v=pearson_all[j]; else if(idx==8) v=pearson_delta_all[j];
        else if(idx==9) v=pearson_delta_de20[j]; else if(idx==10) v=pearson_delta_de50[j];
        else if(idx==11) v=pearson_delta_de100[j]; return v+0;
      }

      END{
        print "==================================================================";
        printf " Final statistics for %s (Genes=%s, %d runs: train+eval)\n", cell_type, gene_count, num_runs;
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

        # CSV header & row
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

        header="Method";
        for(i=1;i<=11;i++){ header = header "," metric_names[i] " (mean±std)" }
        for(r=1;r<=num_runs;r++){
          for(i=1;i<=11;i++){ header = header ",Run" r " " metric_names[i] }
        }

        row=method;
        for(i=1;i<=11;i++){
          ms=mean_std(i); split(ms,parts,"|");
          row = row sprintf(",%.4f±%.4f", parts[1], parts[2]);
        }
        for(r=0;r<num_runs;r++){
          for(i=1;i<=11;i++){ row = row sprintf(",%.4f", val(i,r)) }
        }

        print header > csv_path;   # overwrite per gene_num (fresh file)
        print row   >> csv_path;   # then append the row
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    ' <<< "$all_outputs"

    echo
    echo "--- Finished all ${NUM_RUNS} runs for ${CELL_TYPE}, Genes=${gene_num} ---"
    echo "Combined Log: ${gene_log_file}"
    echo "CSV at: ${csv_file}"

  } | tee -a "${gene_log_file}"

  echo

done

echo "######################################################################"
echo "###   All processing is complete!                                 ###"
echo "######################################################################"
