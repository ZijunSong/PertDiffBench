#!/bin/bash

# Exit on error; print a clear message
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# -------------------- Configuration --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scGen}   # CSV 第一列方法名
# -------------------------------------------------------

# Cell types to evaluate
CELL_TYPES=(
  'B'
  'CD4T'
  'CD8T'
  'CD14+Mono'
  'Dendritic'
  'FCGR3A+Mono'
  'NK'
)

# n_samples map (注意：CD4T=100 按你给出的表)
declare -A SAMPLES_MAP=(
  ['B']=110
  ['CD4T']=100
  ['CD8T']=53
  ['CD14+Mono']=83
  ['Dendritic']=55
  ['FCGR3A+Mono']=132
  ['NK']=54
)

mkdir -p "${LOGDIR}/fig1_task1"

for cell_type in "${CELL_TYPES[@]}"; do
  echo "######################################################################"
  echo "###   Starting to process cell type: $cell_type (${NUM_RUNS} runs total)"
  echo "######################################################################"

  n_samples=${SAMPLES_MAP[$cell_type]}
  [ -z "$n_samples" ] && { echo "No n_samples configured for ${cell_type}"; exit 1; }
  echo "### Using n_samples: $n_samples for this cell type."

  # 基础路径
  train_path="data/fig1/hvg_task1/${cell_type}_train_HVG_1000.h5ad"
  valid_path="data/fig1/hvg_task1/${cell_type}_valid_HVG_1000.h5ad"

  pred_base="samples/fig1/task1/${cell_type}/scgen_1000/"
  mkdir -p "${pred_base}"

  # CSV + 日志文件
  csv_path="${pred_base}/metrics_${METHOD_NAME}_${cell_type}_hvg_1000.csv"
  log_file="${LOGDIR}/fig1_task1/scgen_${cell_type}_hvg_1000.log"

  {
    echo "== $(date '+%F %T') | cell_type=${cell_type} runs=${NUM_RUNS} n_samples=${n_samples} =="

    all_outputs=""

    # 3 次运行：每次输出到 run{i} 子目录，避免覆盖
    for (( i=1; i<=NUM_RUNS; i++ )); do
      echo
      echo "======================"
      echo " Run ${i}/${NUM_RUNS} for ${cell_type}"
      echo "======================"

      run_dir="${pred_base}/run${i}"
      mkdir -p "${run_dir}"

      output=$(
        python scripts/scGen_eval.py \
          --train_data_path "${train_path}" \
          --test_data_path  "${valid_path}"  \
          --model_save_path "checkpoints/scgen/${cell_type}_hvg_1000/run${i}" \
          --out_h5ad   "${run_dir}/${cell_type}_1000_pred_${i}.h5ad" \
          --umap_plot  "${run_dir}/${cell_type}_umap_comparison_${i}.png" \
          --n_samples "${n_samples}" \
          --celltype_to_predict "${cell_type}" 2>&1
      ) || true

      echo "$output"
      all_outputs+="$output\n"
    done

    # 统计并写 CSV（含 mean±std 与每次原始值）
    echo
    echo -e "$all_outputs" | awk -v dataset="${cell_type}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
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

      # 计算 mean|std
      function mean_std(idx,    i,n,s,mu,ss,v) {
        if (idx==1)  { n=c_pds;                 for(i=0;i<n;i++){v=pds[i];                 s+=v} }
        else if(idx==2){ n=c_mae;               for(i=0;i<n;i++){v=mae[i];                 s+=v} }
        else if(idx==3){ n=c_des;               for(i=0;i<n;i++){v=des[i];                 s+=v} }
        else if(idx==4){ n=c_edist;             for(i=0;i<n;i++){v=edist[i];               s+=v} }
        else if(idx==5){ n=c_mmd;               for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
        else if(idx==6){ n=c_r2;                for(i=0;i<n;i++){v=r2[i];                  s+=v} }
        else if(idx==7){ n=c_pearson_all;       for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
        else if(idx==8){ n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
        else if(idx==9){ n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
        else if(idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v} }
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

      # 取第 j 次的原始值（0-based）
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

        # 生成 CSV：Method + 11(mean±std) + 每次原始(3x11)
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
        for (i=1;i<=11;i++) {
          ms = mean_std(i); split(ms, parts, "|");
          row = row sprintf(",%.4f±%.4f", parts[1], parts[2]);
        }
        for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));

        print header > csv_path;
        print row    >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    '

    echo
    echo "--- Finished pipeline for cell type: ${cell_type} ---"
    echo
  } 2>&1 | tee -a "${log_file}"

done

echo "######################################################################"
echo "###   All cell type processing is complete!                        ###"
echo "######################################################################"
