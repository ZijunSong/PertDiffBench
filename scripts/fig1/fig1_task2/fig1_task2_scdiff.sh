#!/bin/bash
# Three (train+eval) runs per dataset; write logs and CSV metrics

# Fail fast + print ERROR on failure
trap 'echo ERROR && exit 1' ERR
set -e

# --------------------
# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=${NAME:-v7.5}
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scDiff}
# --------------------

# Project root (script 位于子目录时向上回溯)
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# Datasets
DATASETS=(
  'random1'
  'random2'
  'random3'
)

# 日志根目录（按数据集分别记录）
RUNLOG_ROOT="${LOGDIR}/perturbation_${NAME}"
mkdir -p "${RUNLOG_ROOT}"

for dataset in "${DATASETS[@]}"; do
  echo "######################################################################"
  echo "###   Starting pipeline for dataset: ${dataset}"
  echo "######################################################################"

  # ==== Data settings for this dataset ====
  dataset_name="fig1_task2_${dataset}"
  train_fname="task2_train_${dataset}_bulkRNAseq_exp.h5ad"
  test_fname="task2_test_${dataset}_bulkRNAseq_exp.h5ad"

  data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
  data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${test_fname}"

  # 结果与日志位置
  DATASET_LOG="${RUNLOG_ROOT}/${dataset}.log"
  OUTDIR_BASE="samples/fig1/task2/${dataset}/scdiff"
  mkdir -p "${OUTDIR_BASE}"

  echo -e "\n==== $(date '+%F %T') | Begin dataset=${dataset} ====\n" | tee -a "${DATASET_LOG}"

  # 收集三次评测输出
  all_outputs=""

  # ==== NUM_RUNS times (train+eval) ====
  for ((i=1; i<=NUM_RUNS; i++)); do
    echo -e "\n======================"           | tee -a "${DATASET_LOG}"
    echo -e " Run ${i}/${NUM_RUNS} : ${dataset}" | tee -a "${DATASET_LOG}"
    echo -e "======================"           | tee -a "${DATASET_LOG}"

    # 跑完整流程（训练+评测）；保留所有stdout/stderr
    output=$(python src/scDiff/main.py \
      --custom_data_path data/fig1/task2 \
      --base configs/scdiff/eval_perturbation.yaml \
      --name "${NAME}" \
      --logdir "${LOGDIR}" \
      --postfix "perturbation_${NAME}" \
      ${OFFLINE_SETTINGS} \
      ${data_settings} 2>&1) || true

    # 打印并写入日志
    echo "${output}" | tee -a "${DATASET_LOG}"

    # 累积到统计文本
    all_outputs+="${output}\n"
  done

  # ==== Stats to console + CSV ====
  CSV_FILE="${OUTDIR_BASE}/metrics_scdiff_${dataset}.csv"
  mkdir -p "$(dirname "${CSV_FILE}")"

  echo -e "\n" | tee -a "${DATASET_LOG}"
  echo -e "${all_outputs}" | awk -v dataset="${dataset}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${CSV_FILE}" '
    # -------- Capture 11 metrics from eval output --------
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

    # -------- helpers --------
    function mean_std(idx,   i,n,s,mu,ss,v){
      if (idx==1){n=c_pds;                  for(i=0;i<n;i++){v=pds[i];                 s+=v}}
      else if(idx==2){n=c_mae;              for(i=0;i<n;i++){v=mae[i];                 s+=v}}
      else if(idx==3){n=c_des;              for(i=0;i<n;i++){v=des[i];                 s+=v}}
      else if(idx==4){n=c_edist;            for(i=0;i<n;i++){v=edist[i];               s+=v}}
      else if(idx==5){n=c_mmd;              for(i=0;i<n;i++){v=mmd[i];                 s+=v}}
      else if(idx==6){n=c_r2;               for(i=0;i<n;i++){v=r2[i];                  s+=v}}
      else if(idx==7){n=c_pearson_all;      for(i=0;i<n;i++){v=pearson_all[i];         s+=v}}
      else if(idx==8){n=c_pearson_delta_all;for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v}}
      else if(idx==9){n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i]; s+=v}}
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

      # ---------- CSV with mean±std + per-run ----------
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
  ' | tee -a "${DATASET_LOG}"

  echo -e "\n--- Finished pipeline for dataset: ${dataset} ---\n" | tee -a "${DATASET_LOG}"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                          ###"
echo "######################################################################"
