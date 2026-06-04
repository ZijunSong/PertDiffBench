#!/bin/bash
# 3 train+eval runs; stats to log/csv

set -e
trap "echo ERROR && exit 1" ERR

# -------------------- Config --------------------
NUM_GENES="1000"
NUM_RUNS=3
METHOD_NAME="scDiffusion" # CSV firstcols name
N_SAMPLES=100 # each run countamount (and )

DATASETS=(
  'mix2'
  'mix3'
  'mix4'
  'mix5'
  'mix6'
  'mix7'
)

# directory (can need )
VAE_CKPT_ROOT="checkpoints/scdiffusion/vae_checkpoint"
DIFF_CKPT_ROOT="checkpoints/scdiffusion/diffusion_checkpoint"
CLS_CKPT_ROOT="checkpoints/scdiffusion/classifier_checkpoint/2-classifier"
SAMPLE_ROOT="samples/fig1/task3"
DATA_ROOT="data/fig1/hvg_task3"

# -------------------- Main loop --------------------
for dataset in "${DATASETS[@]}"; do
  echo "######################################################################"
  echo "###   Starting full pipeline for dataset: ${dataset}"
  echo "######################################################################"

  TRAIN_H5="${DATA_ROOT}/${dataset}_train_HVG_${NUM_GENES}.h5ad"
  TEST_H5="${DATA_ROOT}/${dataset}_test_HVG_${NUM_GENES}.h5ad"

  # output dir ( + evalfile + CSV)
  OUT_BASE="${SAMPLE_ROOT}/${dataset}/scdiffusion_1000/"
  mkdir -p "${OUT_BASE}"
  LOG_FILE="${OUT_BASE}/pipeline_${dataset}.log"
  : > "${LOG_FILE}"

  all_outputs=""

  # -------- 3 (train+eval) --------
  for (( run_idx=1; run_idx<=NUM_RUNS; run_idx++ )); do
    echo -e "\n======================"       | tee -a "${LOG_FILE}"
    echo -e " Run ${run_idx}/${NUM_RUNS} " | tee -a "${LOG_FILE}"
    echo -e "======================"       | tee -a "${LOG_FILE}"

    # directory ( run)
    VAE_DIR="${VAE_CKPT_ROOT}/${dataset}_${NUM_GENES}/run${run_idx}"
    DIFF_DIR="${DIFF_CKPT_ROOT}/${dataset}_${NUM_GENES}/run${run_idx}"
    CLS_DIR="${CLS_CKPT_ROOT}/${dataset}_${NUM_GENES}/run${run_idx}"
    SAMPLE_DIR="${OUT_BASE}/run${run_idx}"
    mkdir -p "${VAE_DIR}" "${DIFF_DIR}" "${CLS_DIR}" "${SAMPLE_DIR}"

    # filename ( andoriginaltrain output )
    VAE_PT="${VAE_DIR}/model_seed=0_step=9999.pt"
    DIFF_PT="${DIFF_DIR}/my_diffusion/model010000.pt"
    CLS_PT="${CLS_DIR}/model009999.pt"

    # --- Step 1: Train VAE ---
    echo -e "\n--- Step 1: Training VAE (run=${run_idx}) ---" | tee -a "${LOG_FILE}"
    pushd src/scDiffusion/VAE >/dev/null
    python VAE_train.py \
      --data_dir "../../../${TRAIN_H5}" \
      --num_genes "$NUM_GENES" \
      --state_dict ../../../checkpoints/annotation_model_v1 \
      --save_dir "../../../${VAE_DIR}" 2>&1 | tee -a "../../../${LOG_FILE}"
    popd >/dev/null

    # --- Step 2: Train Diffusion ---
    echo -e "\n--- Step 2: Training Diffusion (run=${run_idx}) ---" | tee -a "${LOG_FILE}"
    pushd src/scDiffusion >/dev/null
    python cell_train.py \
      --data_dir "../../${TRAIN_H5}" \
      --vae_path "../../${VAE_PT}" \
      --save_dir "../../${DIFF_DIR}" 2>&1 | tee -a "../../${LOG_FILE}"
    popd >/dev/null

    # --- Step 3: Train Classifier ---
    echo -e "\n--- Step 3: Training Classifier (run=${run_idx}) ---" | tee -a "${LOG_FILE}"
    pushd src/scDiffusion >/dev/null
    python classifier_train.py \
      --data_dir "../../${TRAIN_H5}" \
      --vae_path "../../${VAE_PT}" \
      --model_path "../../${CLS_DIR}" 2>&1 | tee -a "../../${LOG_FILE}"
    popd >/dev/null

    # --- Step 4: Sampling & Evaluation ---
    echo -e "\n--- Step 4: Sampling & Evaluation (run=${run_idx}) ---" | tee -a "${LOG_FILE}"
    pushd src/scDiffusion >/dev/null
    output=$(python classifier_sample.py \
      --num_samples "${N_SAMPLES}" \
      --train-data-path "../../${TRAIN_H5}" \
      --model_path "../../${DIFF_PT}" \
      --classifier_path "../../${CLS_PT}" \
      --ae_dir "../../${VAE_PT}" \
      --num_gene "$NUM_GENES" \
      --sample_dir "../../${SAMPLE_DIR}" \
      --out_h5ad "../../${SAMPLE_DIR}/synthetic_ifn_run${run_idx}.h5ad" \
      --umap_plot "../../${SAMPLE_DIR}/umap_comparison_run${run_idx}.png" \
      --init_cell_path "../../${TEST_H5}" 2>&1) || true
    popd >/dev/null

    echo "$output" | tee -a "${LOG_FILE}"
    all_outputs+="$output\n"
  done

  # -------- stats + CSV --------
  CSV_FILE="${OUT_BASE}/metrics_${METHOD_NAME}_${dataset}_gene_${NUM_GENES}.csv"

  echo -e "\n" | tee -a "${LOG_FILE}"
  echo -e "$all_outputs" | awk -v dataset="$dataset" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$CSV_FILE" '
    # (11items)
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

      # CSV: header + countvalue (mean±std + each runoriginalvalue)
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

  echo -e "\n--- Finished pipeline for dataset: ${dataset} ---\n" | tee -a "${LOG_FILE}"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                        ###"
echo "######################################################################"
