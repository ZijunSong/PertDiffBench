#!/bin/bash
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
export PYTHONUNBUFFERED=1
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# =================== Config ===================
SEEDS=( "123" "345" "567" )
NUM_RUNS="${NUM_RUNS:-3}"
GENE_SIZE="${GENE_SIZE:-3000}"
OUTPUT_DIM="${OUTPUT_DIM:-3000}"
METHOD_NAME="${METHOD_NAME:-Squidiff}"

LOGROOT="${LOGROOT:-logs/squidiff}" # only Python inside , and 
CKPT_ROOT="${CKPT_ROOT:-checkpoints/squidiff}"
SAMPLES_ROOT="${SAMPLES_ROOT:-samples/fig2/task1}"

# =================== Main =====================
for seed in "${SEEDS[@]}"; do
  dataset_name="seed${seed}"

  echo "######################################################################"
  echo "###   Starting to process dataset: $dataset_name"
  echo "######################################################################"

  TRAIN_DATA="data/fig2/task1_unseen_pert/${dataset_name}_control_train.h5ad"
  TEST_DATA="data/fig2/task1_unseen_pert/${dataset_name}_control_test.h5ad"

  N_SAMPLES="$(max_n_samples_paired "${TEST_DATA}")"

  # for statsevaloutput
  all_outputs=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    # ---- directory ( run)----
    RUN_CKPT_DIR="${CKPT_ROOT}/${dataset_name}/run${i}"
    RUN_OUT_DIR="${SAMPLES_ROOT}/${dataset_name}/squidiff/run${i}"
    mkdir -p "${RUN_CKPT_DIR}" "${RUN_OUT_DIR}"

    # ---- Step 1: Train ( to , outside file)----
    echo -e "\n--- Training model for ${dataset_name} (run ${i}) ---"
    python src/Squidiff/train_squidiff.py \
      --logger_path "${LOGROOT}/fig2_task2_${dataset_name}_run${i}" \
      --data_path "${TRAIN_DATA}" \
      --resume_checkpoint "${RUN_CKPT_DIR}" \
      --gene_size "${GENE_SIZE}" \
      --output_dim "${OUTPUT_DIM}"

    echo "--- Training for ${dataset_name} (run ${i}) complete. ---"

    # ---- Step 2: Evaluate ( when stdout/stderr to , when for stats)----
    echo -e "\n--- Evaluating (sampling) for ${dataset_name} (run ${i}) ---"

    PRED_H5AD="${RUN_OUT_DIR}/synthetic_ifn_run_${i}.h5ad"
    UMAP_PNG="${RUN_OUT_DIR}/umap_comparison_${i}.png"
    MODEL_PT="${RUN_CKPT_DIR}/model.pt"

    # eval , Traceback willdirectlyin ; when after run (keep )
    output="$(
      python src/Squidiff/sample_squidiff.py \
        --model_path "${MODEL_PT}" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${OUTPUT_DIM}" \
        --out_h5ad "${PRED_H5AD}" \
        --n_samples "${N_SAMPLES}" \
        --umap_plot "${UMAP_PNG}" \
        --train_data_path "${TRAIN_DATA}" \
        --data_path "${TEST_DATA}" 2>&1 | { if [ -t 1 ]; then tee /dev/tty; else cat; fi; }
    )" || true

    # when , hereonlyfor stats
    all_outputs+="${output}\n"
  done

  # ---- Step 3: stats + CSV output ( value± toand value)----
  CSV_FILE="${SAMPLES_ROOT}/${dataset_name}/squidiff/metrics_squidiff_${dataset_name}.csv"
  mkdir -p "$(dirname "${CSV_FILE}")"

  echo -e "\n"
  echo -e "${all_outputs}" | awk -v dataset="${dataset_name}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${CSV_FILE}" '
    # -------- Capture metrics (11) --------
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

    function mean_std(idx,   i,n,s,mu,ss,v){
      if (idx==1){n=c_pds;                   for(i=0;i<n;i++){v=pds[i];                 s+=v}}
      else if(idx==2){n=c_mae;               for(i=0;i<n;i++){v=mae[i];                 s+=v}}
      else if(idx==3){n=c_des;               for(i=0;i<n;i++){v=des[i];                 s+=v}}
      else if(idx==4){n=c_edist;             for(i=0;i<n;i++){v=edist[i];               s+=v}}
      else if(idx==5){n=c_mmd;               for(i=0;i<n;i++){v=mmd[i];                 s+=v}}
      else if(idx==6){n=c_r2;                for(i=0;i<n;i++){v=r2[i];                  s+=v}}
      else if(idx==7){n=c_pearson_all;       for(i=0;i<n;i++){v=pearson_all[i];         s+=v}}
      else if(idx==8){n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v}}
      else if(idx==9){n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v}}
      else if(idx==10){n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v}}
      else if(idx==11){n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v}}
      mu=(n>0)? s/n : 0
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
        ss += (v-mu)*(v-mu)
      }
      return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0"
    }
    function val(idx, j, v){
      if (idx==1) v=pds[j];
      else if(idx==2) v=mae[j];
      else if(idx==3) v=des[j];
      else if(idx==4) v=edist[j];
      else if(idx==5) v=mmd[j];
      else if(idx==6) v=r2[j];
      else if(idx==7) v=pearson_all[j];
      else if(idx==8) v=pearson_delta_all[j];
      else if(idx==9) v=pearson_delta_de20[j];
      else if(idx==10) v=pearson_delta_de50[j];
      else if(idx==11) v=pearson_delta_de100[j];
      return v
    }
    function print_stat(name, data, count,   i,s,mu,ss,std){
      if (count>0){
        for(i=0;i<count;i++) s+=data[i]
        mu=s/count
        for(i=0;i<count;i++) ss+=(data[i]-mu)^2
        std=(count>1)?sqrt(ss/(count-1)):0
        printf "%-40s: %.4f ± %.4f\n", name, mu, std
      } else {
        printf "%-40s: N/A (No data collected)\n", name
      }
    }

    END {
      print "==================================================================";
      printf " Final statistics for dataset %s (%d runs: train+eval)\n", dataset, num_runs;
      print "==================================================================";

      print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
      print_stat("Mean Absolute Error (MAE)",       mae, c_mae);
      print_stat("Differential Expression Score (DES)", des, c_des);
      print "----------------------------------------";
      print_stat("E-Distance",                        edist, c_edist);
      print_stat("Maximum Mean Discrepancy (MMD)",    mmd,  c_mmd);
      print_stat("R-squared (R2)",                    r2,   c_r2);
      print "----------------------------------------";
      print_stat("Pearson (all genes)",               pearson_all,         c_pearson_all);
      print_stat("Pearson Delta (all genes)",         pearson_delta_all,   c_pearson_delta_all);
      print_stat("Pearson Delta (top 20 DE genes)",   pearson_delta_de20,  c_pearson_delta_de20);
      print_stat("Pearson Delta (top 50 DE genes)",   pearson_delta_de50,  c_pearson_delta_de50);
      print_stat("Pearson Delta (top 100 DE genes)",  pearson_delta_de100, c_pearson_delta_de100);
      print "==================================================================\n";

      # ---------- CSV (mean±std + per-run) ----------
      metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES";
      metric_names[4]="E-Distance"; metric_names[5]="MMD"; metric_names[6]="R2";
      metric_names[7]="Pearson (all genes)";
      metric_names[8]="Pearson Delta (all genes)";
      metric_names[9]="Pearson Delta (top 20 DE genes)";
      metric_names[10]="Pearson Delta (top 50 DE genes)";
      metric_names[11]="Pearson Delta (top 100 DE genes)";

      header="Method";
      for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
      for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

      row=method;
      for(i=1;i<=11;i++){
        ms=mean_std(i); split(ms, parts, "|");
        row=row sprintf(",%.4f±%.4f", parts[1], parts[2]);
      }
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.4f", val(i,r));

      print header > csv_path;
      print row    >> csv_path;
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished pipeline for dataset: ${dataset_name} ---\n"
done

echo "######################################################################"
echo "###   All datasets processing is complete!                         ###"
echo "######################################################################"
