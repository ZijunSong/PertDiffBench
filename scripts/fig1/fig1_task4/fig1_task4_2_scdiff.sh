#!/bin/bash

# Exit on error and print a clear message
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# --------------------
# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=${NAME:-v7.5}
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scDiff}   # CSV 第一列方法名
# --------------------

# Project root
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# Dataset prefixes
PREFIXES=( "ACTA2" "B2M" )

for prefix in "${PREFIXES[@]}"; do
  dataset_name="task4_${prefix}_control_to_ifn"
  train_fname="task4_${prefix}_control_to_ifn.h5ad"
  test_fname="task4_${prefix}_control_to_coculture.h5ad"

  echo "######################################################################"
  echo "###   Starting pipeline for dataset: ${dataset_name}"
  echo "######################################################################"

  # scDiff 的 hydra 参数
  data_settings=(
    "data.params.train.params.dataset=${dataset_name}"
    "data.params.train.params.fname=${train_fname}"
    "data.params.test.params.dataset=${dataset_name}"
    "data.params.test.params.fname=${test_fname}"
  )

  # 仅保留 CSV 产物目录
  CSV_DIR="samples/fig1/task4_2/${prefix}/scdiff"
  SUMMARY_CSV="${CSV_DIR}/${dataset_name}.csv"
  mkdir -p "${CSV_DIR}"

  # 聚合所有 run 的原始输出，供 AWK 解析
  ALL_OUT_TMP="$(mktemp)"

  for ((i=1; i<=NUM_RUNS; i++)); do
    echo -e "\n--- Running iteration ${i}/${NUM_RUNS} for ${dataset_name} ---"
    # 关键点：用 tee 既把输出打到 stdout（让 nohup 接走），又写入聚合临时文件
    if python src/scDiff/main.py \
        --custom_data_path data/fig1/task4 \
        --base configs/scdiff/eval_perturbation.yaml \
        --name "${NAME}" \
        --logdir "${LOGDIR}" \
        --postfix "perturbation_${NAME}" \
        ${OFFLINE_SETTINGS} \
        "${data_settings[@]}" \
        2>&1 | tee -a "${ALL_OUT_TMP}"
    then
      # run 之间加空行，便于正则稳健
      echo >> "${ALL_OUT_TMP}"
    else
      echo "[ERROR] Run ${i} failed." >&2
      exit 1   # 想遇错继续可改成 continue
    fi
  done

  echo -e "\n===== Final statistics for ${dataset_name} (${NUM_RUNS} runs) ====="

  # 依赖评估脚本打印以下 11 行标签：
  # 1) Perturbation Discrimination Score (PDS): <num>
  # 2) Mean Absolute Error (MAE): <num>
  # 3) Differential Expression Score (DES): <num>
  # 4) E-Distance: <num>
  # 5) Maximum Mean Discrepancy (MMD): <num>
  # 6) R-squared (R2): <num>
  # 7) Pearson (all genes): <num>
  # 8) Pearson Delta (all genes): <num>
  # 9) Pearson Delta (top 20 DE genes): <num>
  # 10) Pearson Delta (top 50 DE genes): <num>
  # 11) Pearson Delta (top 100 DE genes): <num>

  awk -v ds="${dataset_name}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${SUMMARY_CSV}" '
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
    /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
    /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
    /^E-?Distance:/                             { ed[c_ed++]   = to_num($NF) }
    /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = to_num($NF) }
    /R-?squared \(?R2\)?:/                      { r2[c_r2++]   = to_num($NF) }
    /Pearson \(all genes\):/                    { p_all[c_p_all++]   = to_num($NF) }
    /Pearson Delta \(all genes\):/              { pd_all[c_pd_all++] = to_num($NF) }
    /Pearson Delta \(top 20 DE genes\):/        { pd20[c_pd20++]     = to_num($NF) }
    /Pearson Delta \(top 50 DE genes\):/        { pd50[c_pd50++]     = to_num($NF) }
    /Pearson Delta \(top 100 DE genes\):/       { pd100[c_pd100++]   = to_num($NF) }

    function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n:0 }
    function std(a,n, mu,s,i){ if(n<=1)return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }

    function mean_std(idx,   n,mu,sd){
      if(idx==1){ n=c_pds;   mu=mean(pds,n);   sd=std(pds,n) }
      else if(idx==2){ n=c_mae;  mu=mean(mae,n);  sd=std(mae,n) }
      else if(idx==3){ n=c_des;  mu=mean(des,n);  sd=std(des,n) }
      else if(idx==4){ n=c_ed;   mu=mean(ed,n);   sd=std(ed,n) }
      else if(idx==5){ n=c_mmd;  mu=mean(mmd,n);  sd=std(mmd,n) }
      else if(idx==6){ n=c_r2;   mu=mean(r2,n);   sd=std(r2,n) }
      else if(idx==7){ n=c_p_all;   mu=mean(p_all,n);   sd=std(p_all,n) }
      else if(idx==8){ n=c_pd_all;  mu=mean(pd_all,n);  sd=std(pd_all,n) }
      else if(idx==9){ n=c_pd20;    mu=mean(pd20,n);    sd=std(pd20,n) }
      else if(idx==10){ n=c_pd50;   mu=mean(pd50,n);    sd=std(pd50,n) }
      else if(idx==11){ n=c_pd100;  mu=mean(pd100,n);   sd=std(pd100,n) }
      return sprintf("%.6f±%.6f", mu, sd)
    }

    function val(idx, r,    v){
      if(idx==1) v=pds[r];
      else if(idx==2) v=mae[r];
      else if(idx==3) v=des[r];
      else if(idx==4) v=ed[r];
      else if(idx==5) v=mmd[r];
      else if(idx==6) v=r2[r];
      else if(idx==7) v=p_all[r];
      else if(idx==8) v=pd_all[r];
      else if(idx==9) v=pd20[r];
      else if(idx==10) v=pd50[r];
      else if(idx==11) v=pd100[r];
      return (v=="") ? 0 : v;
    }

    END{
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

      # Header（覆盖写入）
      header="Dataset,Method";
      for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
      for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

      # Row
      row=ds "," method;
      for(i=1;i<=11;i++) row=row "," mean_std(i);

      # runs（按 0..num_runs-1）
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

      print header > csv_path;
      print row    >> csv_path;

      # 打印可读汇总到 stdout（由 nohup 接管）
      printf "%-40s: %s\n", "Perturbation Discrimination (PDS)", mean_std(1);
      printf "%-40s: %s\n", "Mean Absolute Error (MAE)",        mean_std(2);
      printf "%-40s: %s\n", "Differential Expression Score (DES)", mean_std(3);
      print  "----------------------------------------";
      printf "%-40s: %s\n", "E-Distance", mean_std(4);
      printf "%-40s: %s\n", "Maximum Mean Discrepancy (MMD)", mean_std(5);
      printf "%-40s: %s\n", "R-squared (R2)", mean_std(6);
      print  "----------------------------------------";
      printf "%-40s: %s\n", "Pearson (all genes)", mean_std(7);
      printf "%-40s: %s\n", "Pearson Delta (all genes)", mean_std(8);
      printf "%-40s: %s\n", "Pearson Delta (top 20 DE genes)", mean_std(9);
      printf "%-40s: %s\n", "Pearson Delta (top 50 DE genes)", mean_std(10);
      printf "%-40s: %s\n", "Pearson Delta (top 100 DE genes)", mean_std(11);
    }
  ' "${ALL_OUT_TMP}"

  rm -f "${ALL_OUT_TMP}"
  echo -e "\n--- Finished pipeline for dataset: ${dataset_name} ---\n"
done

echo "######################################################################"
echo "###   All dataset processing is complete!                          ###"
echo "######################################################################"
