#!/bin/bash
# (train+eval), each runstandalone checkpoint; stats log and csv

set -e
trap "echo ERROR && exit 1" ERR

# ------------ ------------
NUM_GENES=1000
NUM_RUNS=3
METHOD_NAME="SquiDiff" # CSV firstcols name
N_SAMPLES=100

MIX_TYPES=('mix2' 'mix3' 'mix4' 'mix5' 'mix6' 'mix7')

DATA_ROOT="data/fig1/hvg_task3"
CKPT_ROOT="checkpoints/squidiff"
LOG_ROOT="logs/squidiff"
SAMPLE_ROOT="samples/fig1/task3"
mkdir -p "${CKPT_ROOT}" "${LOG_ROOT}" "${SAMPLE_ROOT}"

# ------------ ------------
for mix_type in "${MIX_TYPES[@]}"; do
  echo "######################################################################"
  echo "###   Starting to process mix type (Task 3): ${mix_type}"
  echo "######################################################################"

  TRAIN_H5="${DATA_ROOT}/${mix_type}_train_HVG_${NUM_GENES}.h5ad"
  TEST_H5="${DATA_ROOT}/${mix_type}_test_HVG_${NUM_GENES}.h5ad"

  # data level output dir
  OUT_BASE="${SAMPLE_ROOT}/${mix_type}/squidiff_1000"
  mkdir -p "${OUT_BASE}"
  LOG_FILE="${OUT_BASE}/pipeline_${mix_type}.log"
  : > "${LOG_FILE}"

  # "eval " output, AWK stats
  all_outputs=""

  # -------- (train+eval) --------
  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo -e "\n======================"       | tee -a "${LOG_FILE}"
    echo -e " Run ${i}/${NUM_RUNS} "        | tee -a "${LOG_FILE}"
    echo -e "======================"       | tee -a "${LOG_FILE}"

    # run path
    RUN_CKPT_DIR="${CKPT_ROOT}/${mix_type}_${NUM_GENES}/run${i}"
    RUN_LOG_DIR="${LOG_ROOT}/${mix_type}_train_HVG_${NUM_GENES}/run${i}"
    RUN_SAMPLE_DIR="${OUT_BASE}/run${i}"
    mkdir -p "${RUN_CKPT_DIR}" "${RUN_LOG_DIR}" "${RUN_SAMPLE_DIR}"

    # trainafter filename ( train , here)
    MODEL_PT="${RUN_CKPT_DIR}/model.pt"

    # --- train ---
    echo -e "\n--- Training model for ${mix_type} (run=${i}) ---" | tee -a "${LOG_FILE}"
    python src/Squidiff/train_squidiff.py \
      --logger_path "${RUN_LOG_DIR}" \
      --data_path   "${TRAIN_H5}" \
      --resume_checkpoint "${RUN_CKPT_DIR}" \
      --gene_size   "${NUM_GENES}" \
      --output_dim  "${NUM_GENES}" 2>&1 | tee -a "${LOG_FILE}"

    echo "--- Training for ${mix_type} (run=${i}) complete. ---" | tee -a "${LOG_FILE}"

    # --- /eval ---
    echo -e "\n--- Sampling & Evaluation for ${mix_type} (run=${i}) ---" | tee -a "${LOG_FILE}"
    # Ensureon directoryexist
    mkdir -p "${RUN_SAMPLE_DIR}"

    output=$(python src/Squidiff/sample_squidiff.py \
      --model_path "${MODEL_PT}" \
      --gene_size  "${NUM_GENES}" \
      --output_dim "${NUM_GENES}" \
      --out_h5ad   "${RUN_SAMPLE_DIR}/synthetic_ifn_run_${i}.h5ad" \
      --train_data_path "${TRAIN_H5}" \
      --n_samples  "${N_SAMPLES}" \
      --umap_plot  "${RUN_SAMPLE_DIR}/umap_comparison_${i}.png" \
      --data_path  "${TEST_H5}" 2>&1) || true

    echo "$output" | tee -a "${LOG_FILE}"
    all_outputs+="$output\n"
  done

  # -------- stats + CSV --------
  echo -e "\n" | tee -a "${LOG_FILE}"
  CSV_FILE="${OUT_BASE}/metrics_${METHOD_NAME}_${mix_type}_gene_${NUM_GENES}.csv"

  echo -e "$all_outputs" | awk -v dataset="${mix_type}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${CSV_FILE}" '
    # 11 items 
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

    # mean|std
    function mean_std(idx,   i,n,s,mu,ss,v) {
      if (idx==1){ n=c_pds;                for(i=0;i<n;i++){v=pds[i];                s+=v} }
      else if(idx==2){ n=c_mae;            for(i=0;i<n;i++){v=mae[i];                s+=v} }
      else if(idx==3){ n=c_des;            for(i=0;i<n;i++){v=des[i];                s+=v} }
      else if(idx==4){ n=c_edist;          for(i=0;i<n;i++){v=edist[i];              s+=v} }
      else if(idx==5){ n=c_mmd;            for(i=0;i<n;i++){v=mmd[i];                s+=v} }
      else if(idx==6){ n=c_r2;             for(i=0;i<n;i++){v=r2[i];                 s+=v} }
      else if(idx==7){ n=c_pearson_all;    for(i=0;i<n;i++){v=pearson_all[i];        s+=v} }
      else if(idx==8){ n=c_pearson_delta_all;   for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
      else if(idx==9){ n=c_pearson_delta_de20;  for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
      else if(idx==10){ n=c_pearson_delta_de50; for(i=0;i<n;i++){v=pearson_delta_de50[i];  s+=v} }
      else if(idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
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

    # j (0-based)countvalue
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

    function print_stat(name, arr, cnt,   i,s,mu,ss,std){
      if (cnt>0){
        for(i=0;i<cnt;i++) s+=arr[i];
        mu=s/cnt;
        for(i=0;i<cnt;i++) ss+=(arr[i]-mu)^2;
        std=(cnt>1)?sqrt(ss/(cnt-1)):0;
        printf "%-40s: %.4f ± %.4f\n", name, mu, std;
      } else {
        printf "%-40s: N/A (No data collected)\n", name;
      }
    }

    END{
      # 
      print "==================================================================";
      printf " Final statistics for dataset %s (%d runs)\n", dataset, num_runs;
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

      # CSV: header + countvalue (11 mean±std + 3*11 value)
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
      for(i=1;i<=11;i++){ header=header "," metric_names[i] " (mean±std)" }
      for(r=1;r<=num_runs;r++){
        for(i=1;i<=11;i++){ header=header ",Run" r " " metric_names[i] }
      }

      row=method;
      for(i=1;i<=11;i++){
        ms=mean_std(i); split(ms, parts, "|");
        row=row sprintf(",%.4f±%.4f", parts[1], parts[2]);
      }
      for(r=0;r<num_runs;r++){
        for(i=1;i<=11;i++){
          row=row sprintf(",%.4f", val(i, r));
        }
      }

      print header > csv_path;
      print row    >> csv_path;
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  ' | tee -a "${LOG_FILE}"

  echo -e "\n--- Finished pipeline for mix type: ${mix_type} ---\n" | tee -a "${LOG_FILE}"
done

echo "######################################################################"
echo "###   All mix-type processing is complete!                        ###"
echo "######################################################################"
