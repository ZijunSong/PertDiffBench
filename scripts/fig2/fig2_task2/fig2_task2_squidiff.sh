#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# -------------------- Config --------------------
TARGET_CELL_TYPES=( "B" "NK" )
GENE_SIZE="${GENE_SIZE:-6998}"
NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-Squidiff}"

# n_samples ( and currenteval can )
source "scripts/lib/max_n_samples.sh"
declare -A SAMPLES_MAP=()
for _ct in "${TARGET_CELL_TYPES[@]}"; do
  _test="../../data/fig2/task2/task2_test_${_ct}_exp.h5ad"
  SAMPLES_MAP["${_ct}"]="$(max_n_samples_paired "${_test}")"
done

echo "Changing directory to src/Squidiff..."
cd src/Squidiff

# -------------------- Main ----------------------
for cell_type in "${TARGET_CELL_TYPES[@]}"; do
  n_samples="${SAMPLES_MAP[$cell_type]}"
  : "${n_samples:?No n_samples configured for ${cell_type}}"

  echo "######################################################################"
  echo "###   Target cell type: ${cell_type} | genes=${GENE_SIZE} | runs=${NUM_RUNS}"
  echo "######################################################################"

  # output dir (and --out_h5ad level directory )
  sample_base="../../samples/fig2/task2/${cell_type}/squidiff"
  ckpt_base="../../checkpoints/fig2/task2/pretrain_CD4T/squidiff"
  mkdir -p "${sample_base}" "${ckpt_base}"

  all_outputs=""

  # -------- 3x runs: Train + Inference --------
  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for ${cell_type}"
    echo "======================"

    run_ckpt_dir="${ckpt_base}/run${i}"
    run_sample_dir="${sample_base}/run${i}"
    mkdir -p "${run_ckpt_dir}" "${run_sample_dir}"

    # train (each run run standalonedirectory, )
    echo
    echo "--- Training on CD4T data [run ${i}] ---"
    python train_squidiff.py \
      --logger_path "../../logs/squidiff/${cell_type}_train_CD4T_g${GENE_SIZE}_run${i}" \
      --data_path "../../data/fig1/raw_task1/task1_train_CD4T_exp.h5ad" \
      --resume_checkpoint "${run_ckpt_dir}" \
      --gene_size "${GENE_SIZE}" \
      --output_dim "${GENE_SIZE}"

    model_path="${run_ckpt_dir}/model.pt" # train name , here

    # /eval
    echo
    echo "--- Inference & evaluation on ${cell_type} [run ${i}] ---"
    output=$(
      python sample_squidiff.py \
        --model_path "${model_path}" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${GENE_SIZE}" \
        --out_h5ad "${run_sample_dir}/synthetic_ifn_run_${i}.h5ad" \
        --n_samples "${n_samples}" \
        --umap_plot "${run_sample_dir}/umap_comparison_${i}.svg" \
        --train_data_path "../../data/fig1/raw_task1/task1_train_CD4T_exp.h5ad" \
        --data_path "../../data/fig1/raw_task1/task1_valid_${cell_type}_exp.h5ad" 2>&1
    ) || true

    echo "$output"
    # and 
    all_outputs+="${output}"$'\n'
  done

  # -------- stats + CSV --------
  csv_path="${sample_base}/metrics_${METHOD_NAME}_${cell_type}_g${GENE_SIZE}.csv"

  echo
  printf "%s" "$all_outputs" | awk -v dataset="${cell_type}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
    # 11 items ( eval count $NF)
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
      if (idx==1)  { n=c_pds;                    for(i=0;i<n;i++){v=pds[i];                 s+=v} }
      else if(idx==2){ n=c_mae;                  for(i=0;i<n;i++){v=mae[i];                 s+=v} }
      else if(idx==3){ n=c_des;                  for(i=0;i<n;i++){v=des[i];                 s+=v} }
      else if(idx==4){ n=c_edist;                for(i=0;i<n;i++){v=edist[i];               s+=v} }
      else if(idx==5){ n=c_mmd;                  for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
      else if(idx==6){ n=c_r2;                   for(i=0;i<n;i++){v=r2[i];                  s+=v} }
      else if(idx==7){ n=c_pearson_all;          for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
      else if(idx==8){ n=c_pearson_delta_all;    for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
      else if(idx==9){ n=c_pearson_delta_de20;   for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
      else if(idx==10){ n=c_pearson_delta_de50;  for(i=0;i<n;i++){v=pearson_delta_de50[i];  s+=v} }
      else if(idx==11){ n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
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

    function val(idx, j,    v){
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

    function print_stat(name, data, count,    i,s,mu,ss,std) {
      if (count > 0) {
        for (i = 0; i < count; i++) s += data[i];
        mu = s / count;
        for (i = 0; i < count; i++) ss += (data[i] - mu)^2;
        std = (count > 1) ? sqrt(ss / (count - 1)) : 0;
        printf "%-40s: %.4f ± %.4f\n", name, mu, std;
      } else {
        printf "%-40s: N/A (No data)\n", name;
      }
    }

    END {
      print "==================================================================";
      printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
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

      # CSV header: + 11(mean±std) + raw(3x11)
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

      header = "Method";
      for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
      for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];

      row = method;
      for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
      for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) {
        v = val(i, r);
        if (v == "") row = row ",";
        else row = row sprintf(",%.4f", v);
      }

      # CSV ( path)
      print header > csv_path;
      print row    >> csv_path;
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo
  echo "--- Finished pipeline for cell type: ${cell_type} ---"
  echo
done

echo "######################################################################"
echo "###   All cell type processing is complete!                        ###"
echo "######################################################################"
