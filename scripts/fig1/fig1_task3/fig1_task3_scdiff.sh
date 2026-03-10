#!/bin/bash

# 失败即报错退出
set -e
trap "echo ERROR && exit 1" ERR

# --------------------
# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"
NUM_RUNS=3
METHOD_NAME="scDiffusion"   # 写入CSV的“方法名”，可按需修改
# --------------------

# 将 HOMEDIR 设为项目根目录（脚本位于子目录时）
HOMEDIR=$(dirname $(dirname $(realpath $0)))/..
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# 数据集列表
DATASETS=(
  'mix2'
  'mix3'
  'mix4'
  'mix5'
  'mix6'
  'mix7'
)

# 主循环：逐数据集
for dataset in "${DATASETS[@]}"; do
  echo "######################################################################"
  echo "###   Starting pipeline for dataset: $dataset"
  echo "######################################################################"

  OUT_ROOT="samples/fig1/task3/${dataset}/scdiff_1000"
  mkdir -p "${OUT_ROOT}"

  # 构造 data_settings
  dataset_name="fig1_task3_${dataset}"
  train_fname="${dataset}_train_HVG_1000.h5ad"
  test_fname="${dataset}_test_HVG_1000.h5ad"

  data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
  data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${test_fname}"

  # 数据集专属日志/CSV路径
  DATASET_OUT_DIR="${OUT_ROOT}/${dataset}"
  mkdir -p "${DATASET_OUT_DIR}"
  LOG_FILE="${DATASET_OUT_DIR}/pipeline_${dataset}.log"
  CSV_FILE="${OUT_ROOT}/metrics_${METHOD_NAME}_${dataset}_gene_1000.csv"
  : > "${LOG_FILE}"

  # 累积3次运行的stdout
  all_outputs=""

  # 3 次（训练+测评）循环
  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo -e "\n======================" | tee -a "${LOG_FILE}"
    echo -e " Run ${i}/${NUM_RUNS} for ${dataset}" | tee -a "${LOG_FILE}"
    echo -e "======================" | tee -a "${LOG_FILE}"

    # 一次完整（训练+评测）
    output=$(python src/scDiff/main.py \
      --custom_data_path data/fig1/hvg_task3 \
      --base configs/scdiff/eval_perturbation.yaml \
      --name "${NAME}" \
      --logdir "${LOGDIR}" \
      --postfix "perturbation_${NAME}" \
      ${OFFLINE_SETTINGS} \
      ${data_settings} 2>&1) || true

    # 同时打印到控制台与日志
    echo "$output" | tee -a "${LOG_FILE}"
    all_outputs+="$output\n"
  done

  # 统计 + CSV：同时写入控制台与日志
  echo -e "\n" | tee -a "${LOG_FILE}"
  echo -e "$all_outputs" | awk -v dataset="$dataset" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$CSV_FILE" '
    # 抓取 11 项指标
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

    # 求 mean|std（根据指标编号选择数组）
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

    # 取第 j 次（0-based）的数值
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
      # 漂亮打印
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

      # CSV：表头 + 一行数值
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
echo "###   All dataset processing is complete!                         ###"
echo "######################################################################"
