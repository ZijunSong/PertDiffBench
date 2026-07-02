#!/bin/bash
# Three (train+eval) runs per dataset; write logs and CSV metrics
set -e

source "scripts/lib/max_n_samples.sh"

# ================== Configuration ==================
DATASETS=('random1' 'random2' 'random3')
NUM_GENES="2969"
NUM_RUNS=3
CONFIG_FILE="configs/baselines/scrna_ddpm_scrna.yaml"
METHOD_NAME="DDPM"
LOGDIR=${LOGDIR:-logs/fig1_task2}
CKPT_NAME="scrna_ddpm_epoch1000.pt"

mkdir -p "$LOGDIR"

# ================== Main Loop ==================
for dataset in "${DATASETS[@]}"; do
  echo "######################################################################"
  echo "###   Starting pipeline for dataset: $dataset"
  echo "######################################################################"

  train_data_path="data/fig1/task2/task2_train_${dataset}_bulkRNAseq_exp.h5ad"
  valid_data_path="data/fig1/task2/task2_test_${dataset}_bulkRNAseq_exp.h5ad"

  N_SAMPLES="$(max_n_samples_paired "${valid_data_path}")"

  save_dir_base="checkpoints/ddpm/${dataset}"
  sample_dir_base="samples/fig1/task2/${dataset}/scrna_ddpm_scrna"
  mkdir -p "${save_dir_base}" "${sample_dir_base}"

  LOGFILE="${LOGDIR}/${dataset}.log"
  echo -e "\n==== $(date '+%F %T') | Begin ${dataset} ====\n" | tee -a "$LOGFILE"

  # aggregate eval stdout for stats
  all_outputs=""

  for (( run_idx=1; run_idx<=NUM_RUNS; run_idx++ )); do
    export RUN_SEED=$(($run_idx-1))
    echo -e "\n======================" | tee -a "$LOGFILE"
    echo -e " Run ${run_idx}/${NUM_RUNS} for dataset ${dataset}" | tee -a "$LOGFILE"
    echo -e "======================" | tee -a "$LOGFILE"

    save_dir_run="${save_dir_base}/run${run_idx}"
    sample_dir_run="${sample_dir_base}/run${run_idx}"
    mkdir -p "${save_dir_run}" "${sample_dir_run}"

    ckpt_path="${save_dir_run}/${CKPT_NAME}"

    # -------- Step 1: Training --------
    echo -e "\n--- Step 1: Training model (dataset: $dataset, run: $run_idx) ---" | tee -a "$LOGFILE"
    # tee training stdout to log
    python scripts/baseline/train_scrna_ddpm_scrna.py \
      --config "$CONFIG_FILE" \
      --data-path "$train_data_path" \
      --save-weight-dir "$save_dir_run" \
      --gene-nums "$NUM_GENES" 2>&1 | tee -a "$LOGFILE"

    # -------- Step 2: Evaluation for this run --------
    echo -e "\n--- Step 2: Evaluating model (dataset: $dataset, run: $run_idx) ---" | tee -a "$LOGFILE"
    # capture eval output; tee to log
    output=$(python scripts/baseline/eval_scrna_ddpm_scrna.py \
      --config "$CONFIG_FILE" \
      --train-data-path "$train_data_path" \
      --data-path "$valid_data_path" \
      --ckpt "$ckpt_path" \
      --out_h5ad "${sample_dir_run}/synthetic_ifn_run${run_idx}.h5ad" \
      --umap_plot "${sample_dir_run}/umap_comparison_run${run_idx}.png" \
      --n_samples "$N_SAMPLES" \
      --gene-nums "$NUM_GENES" 2>&1) || true

    echo "$output" | tee -a "$LOGFILE"
    all_outputs+="$output\n"
  done

  # -------- Step 3: Stats to console + CSV --------
  csv_file="${sample_dir_base}/metrics_ddpm_${dataset}_gene_${NUM_GENES}.csv"
  mkdir -p "${sample_dir_base}"

  echo -e "\n" | tee -a "$LOGFILE"
  echo -e "$all_outputs" | awk -v dataset="$dataset" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$csv_file" '
    # Capture metrics (11)
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

    # compute mean|std for metric idx
    function mean_std(idx,   i,n,s,mu,ss,v) {
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
      mu=(n>0)?s/n:0;
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
        ss += (v-mu)*(v-mu);
      }
      return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
    }
    # single-run value accessor
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
      return v;
    }
    function print_stat(name, data, count,   i,s,mu,ss,std){
      if (count>0){
        for(i=0;i<count;i++) s+=data[i];
        mu=s/count;
        for(i=0;i<count;i++) ss+=(data[i]-mu)^2;
        std=(count>1)?sqrt(ss/(count-1)):0;
        printf "%-40s: %.4f ± %.4f\n", name, mu, std;
      } else {
        printf "%-40s: N/A (No data collected)\n", name;
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

      # CSV header + row
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
  ' | tee -a "$LOGFILE"

  echo -e "\n--- Finished pipeline for dataset: $dataset ---\n" | tee -a "$LOGFILE"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                          ###"
echo "######################################################################"
