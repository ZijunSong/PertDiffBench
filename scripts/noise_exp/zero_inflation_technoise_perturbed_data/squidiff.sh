#!/bin/bash

# 如果任何命令以非零状态退出，则立即退出脚本。
set -e

# -------------------- 配置 --------------------
CELL_TYPES=('CD4T')

NOISE_LEVELS=('0.3' '0.6' '1.0' '1.6' '2.2')

# 通用参数
GENE_SIZE="${GENE_SIZE:-6998}"
NUM_RUNS="${NUM_RUNS:-3}"

METHOD_NAME="${METHOD_NAME:-Squidiff-${GENE_SIZE}}"   # 写到 CSV 里的方法名
METHOD_DIR="${METHOD_DIR:-squidiff}"                  # 作为路径最后一级的方法目录名

BASE_DATA_DIR="data/add_zero_inflation_output"
BASE_CKPT_DIR="checkpoints/zero_inflation_technoise"
BASE_SAMPLES_DIR="samples/zero_inflation_technoise"

mkdir -p logs

# -------------------- 主循环 --------------------
for cell_type in "${CELL_TYPES[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###  处理细胞类型: $cell_type | 噪声等级: $noise_level"
    echo "######################################################################"

    # 动态构建带噪声的训练 / 验证数据文件路径（注意这里不再硬编码 CD4T）
    train_data_file="${BASE_DATA_DIR}/task1_train_${cell_type}_exp_zeroinflation_strength_${noise_level}.h5ad"
    valid_data_file="${BASE_DATA_DIR}/task1_valid_${cell_type}_exp_zeroinflation_strength_${noise_level}.h5ad"

    # 检查带噪声的训练文件是否存在，如果不存在则跳过该组合
    if [ ! -f "$train_data_file" ]; then
      echo "警告: 未找到训练数据文件 '$train_data_file'。将跳过此组合。"
      continue
    fi

    group_suffix="${cell_type}_noise_${noise_level}"

    # 统一的大路径：只在最后一级用 METHOD_DIR 区分方法
    base_checkpoint_dir="${BASE_CKPT_DIR}/${group_suffix}/${METHOD_DIR}"
    base_samples_dir="${BASE_SAMPLES_DIR}/${group_suffix}/${METHOD_DIR}"
    mkdir -p "${base_checkpoint_dir}" "${base_samples_dir}"

    echo -e "\n--- 正在为 $cell_type (噪声: $noise_level) 进行 ${NUM_RUNS} 次 [训练+采样] ---"

    all_outputs=""

    # ---------- 第 1&2 步：三次训练 + 对应三次采样 ----------
    for (( run_id=1; run_id<=NUM_RUNS; run_id++ )); do
      echo -e "\n================ Run ${run_id}/${NUM_RUNS} | ${cell_type} noise=${noise_level} ================"

      # 每个 run 独立的 checkpoint / samples 子目录
      checkpoint_dir="${base_checkpoint_dir}/run_${run_id}"
      samples_dir="${base_samples_dir}/run_${run_id}"
      mkdir -p "${checkpoint_dir}" "${samples_dir}"

      # --- 训练 ---
      echo -e "\n--- [Run ${run_id}] 训练模型 ---"
      python src/Squidiff/train_squidiff.py \
        --logger_path "logs/squidiff/${cell_type}_train_HVG_${GENE_SIZE}_noise_${noise_level}_run${run_id}" \
        --data_path "${train_data_file}" \
        --resume_checkpoint "${checkpoint_dir}" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${GENE_SIZE}" 2>&1 | tee "logs/train_squidiff_${cell_type}_noise_${noise_level}_run${run_id}.log"

      model_path="${checkpoint_dir}/model.pt"
      if [ ! -f "${model_path}" ]; then
        echo "[ERROR] 训练完成后未找到模型权重: ${model_path}" >&2
        exit 1
      fi
      echo "--- [Run ${run_id}] 训练完成，模型: ${model_path} ---"

      # --- 采样 / 评估（失败不终止整体流水线） ---
      echo -e "\n--- [Run ${run_id}] 开始采样 / 评估 ---"
      output=$(
        python src/Squidiff/sample_squidiff.py \
          --model_path "${model_path}" \
          --gene_size "${GENE_SIZE}" \
          --output_dim "${GENE_SIZE}" \
          --out_h5ad "${samples_dir}/synthetic_ifn_run_${run_id}.h5ad" \
          --train_data_path "${train_data_file}" \
          --n_samples 6 \
          --umap_plot "${samples_dir}/umap_comparison_${run_id}.png" \
          --data_path "${valid_data_file}" 2>&1
      ) || true

      echo "${output}" | tee "logs/sample_squidiff_${cell_type}_noise_${noise_level}_run${run_id}.log"
      all_outputs+="${output}"$'\n'
    done

    # ---------- 第 3 步：AWK 统计 + 写 CSV ----------
    CSV_PATH="${base_samples_dir}/metrics_${group_suffix}.csv"
    mkdir -p "$(dirname "${CSV_PATH}")"

    echo -e "\n--- 使用 AWK 聚合 ${NUM_RUNS} 次运行的指标，并写入 CSV: ${CSV_PATH} ---\n"

    echo "${all_outputs}" | awk \
      -v dataset="$cell_type" \
      -v noise="$noise_level" \
      -v num_runs="$NUM_RUNS" \
      -v method="$METHOD_NAME" \
      -v csv_path="$CSV_PATH" '
      # ---------- 函数定义 ----------
      function print_stat(name, arr, cnt,    i,sum,mean,ssd,sd,tmp){
        if (cnt > 0){
          sum=0
          for(i=0;i<cnt;i++) sum += arr[i]+0
          mean = sum/cnt
          ssd=0
          for(i=0;i<cnt;i++){ tmp = arr[i]-mean; ssd += tmp*tmp }
          sd=(cnt>1)? sqrt(ssd/(cnt-1)) : 0
          printf "%-40s: %.4f ± %.4f\n", name, mean, sd
        } else {
          printf "%-40s: N/A (未收集到数据)\n", name
        }
      }

      function mean_std_str(idx,    i,sum,mean,ssd,sd,cnt,tmp){
        sum=ssd=mean=sd=0; cnt=0
        if(idx==1){ cnt=c_pds;                   for(i=0;i<cnt;i++) sum += pds[i]+0 }
        else if(idx==2){ cnt=c_mae;              for(i=0;i<cnt;i++) sum += mae[i]+0 }
        else if(idx==3){ cnt=c_des;              for(i=0;i<cnt;i++) sum += des[i]+0 }
        else if(idx==4){ cnt=c_edist;            for(i=0;i<cnt;i++) sum += edist[i]+0 }
        else if(idx==5){ cnt=c_mmd;              for(i=0;i<cnt;i++) sum += mmd[i]+0 }
        else if(idx==6){ cnt=c_r2;               for(i=0;i<cnt;i++) sum += r2[i]+0 }
        else if(idx==7){ cnt=c_pearson_all;      for(i=0;i<cnt;i++) sum += pearson_all[i]+0 }
        else if(idx==8){ cnt=c_pearson_delta_all;for(i=0;i<cnt;i++) sum += pearson_delta_all[i]+0 }
        else if(idx==9){ cnt=c_pearson_delta_de20;for(i=0;i<cnt;i++) sum += pearson_delta_de20[i]+0 }
        else if(idx==10){cnt=c_pearson_delta_de50;for(i=0;i<cnt;i++) sum += pearson_delta_de50[i]+0 }
        else if(idx==11){cnt=c_pearson_delta_de100;for(i=0;i<cnt;i++) sum += pearson_delta_de100[i]+0 }

        if(cnt>0){
          mean = sum/cnt
          if(idx==1){ for(i=0;i<cnt;i++){ tmp=pds[i]-mean; ssd+=tmp*tmp } }
          else if(idx==2){ for(i=0;i<cnt;i++){ tmp=mae[i]-mean; ssd+=tmp*tmp } }
          else if(idx==3){ for(i=0;i<cnt;i++){ tmp=des[i]-mean; ssd+=tmp*tmp } }
          else if(idx==4){ for(i=0;i<cnt;i++){ tmp=edist[i]-mean; ssd+=tmp*tmp } }
          else if(idx==5){ for(i=0;i<cnt;i++){ tmp=mmd[i]-mean; ssd+=tmp*tmp } }
          else if(idx==6){ for(i=0;i<cnt;i++){ tmp=r2[i]-mean; ssd+=tmp*tmp } }
          else if(idx==7){ for(i=0;i<cnt;i++){ tmp=pearson_all[i]-mean; ssd+=tmp*tmp } }
          else if(idx==8){ for(i=0;i<cnt;i++){ tmp=pearson_delta_all[i]-mean; ssd+=tmp*tmp } }
          else if(idx==9){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de20[i]-mean; ssd+=tmp*tmp } }
          else if(idx==10){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de50[i]-mean; ssd+=tmp*tmp } }
          else if(idx==11){ for(i=0;i<cnt;i++){ tmp=pearson_delta_de100[i]-mean; ssd+=tmp*tmp } }
          sd = (cnt>1)? sqrt(ssd/(cnt-1)) : 0
          return sprintf("%.4f|%.4f", mean, sd)
        }
        return "0.0000|0.0000"
      }

      function val_idx(idx, r,    v){
        if     (idx==1){ v = (r < c_pds)?pds[r]:"" }
        else if(idx==2){ v = (r < c_mae)?mae[r]:"" }
        else if(idx==3){ v = (r < c_des)?des[r]:"" }
        else if(idx==4){ v = (r < c_edist)?edist[r]:"" }
        else if(idx==5){ v = (r < c_mmd)?mmd[r]:"" }
        else if(idx==6){ v = (r < c_r2)?r2[r]:"" }
        else if(idx==7){ v = (r < c_pearson_all)?pearson_all[r]:"" }
        else if(idx==8){ v = (r < c_pearson_delta_all)?pearson_delta_all[r]:"" }
        else if(idx==9){ v = (r < c_pearson_delta_de20)?pearson_delta_de20[r]:"" }
        else if(idx==10){ v = (r < c_pearson_delta_de50)?pearson_delta_de50[r]:"" }
        else if(idx==11){ v = (r < c_pearson_delta_de100)?pearson_delta_de100[r]:"" }
        return v
      }

      function mean_std_str_wrap(i,   ms,parts){
        ms = mean_std_str(i); split(ms,parts,"|")
        return sprintf("%.4f±%.4f", parts[1]+0, parts[2]+0)
      }

      # ---------- 收集指标 ----------
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

      END{
        print "=================================================================="
        printf " %s (噪声: %s) 的最终统计结果 (%d 次运行)\n", dataset, noise, num_runs
        print "=================================================================="

        print_stat("Perturbation Discrimination (PDS)", pds, c_pds)
        print_stat("Mean Absolute Error (MAE)", mae, c_mae)
        print_stat("Differential Expression Score (DES)", des, c_des)
        print "----------------------------------------"
        print_stat("E-Distance", edist, c_edist)
        print_stat("Maximum Mean Discrepancy (MMD)", mmd, c_mmd)
        print_stat("R-squared (R2)", r2, c_r2)
        print "----------------------------------------"
        print_stat("Pearson (all genes)", pearson_all, c_pearson_all)
        print_stat("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all)
        print_stat("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20)
        print_stat("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50)
        print_stat("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100)

        print "==================================================================\n"

        # -------- 写 CSV：Dataset,Noise,Method + mean±std + 每次运行 --------
        metric_names[1]="PDS"
        metric_names[2]="MAE"
        metric_names[3]="DES"
        metric_names[4]="E-Distance"
        metric_names[5]="MMD"
        metric_names[6]="R2"
        metric_names[7]="Pearson (all genes)"
        metric_names[8]="Pearson Delta (all genes)"
        metric_names[9]="Pearson Delta (top 20 DE genes)"
        metric_names[10]="Pearson Delta (top 50 DE genes)"
        metric_names[11]="Pearson Delta (top 100 DE genes)"

        header = "Dataset,Noise,Method"
        for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)"
        for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i]

        row = dataset "," noise "," method
        for (i=1;i<=11;i++) row = row "," mean_std_str_wrap(i)
        for (r=0;r<num_runs;r++){
          for (i=1;i<=11;i++){
            v = val_idx(i, r)
            if (v == "") row = row ","
            else row = row sprintf(",%.4f", v+0)
          }
        }

        print header > csv_path
        print row    >> csv_path
        close(csv_path)
        printf("CSV written: %s\n", csv_path)
      }
    '

    echo -e "\n--- 完成细胞类型: $cell_type | 噪声等级: $noise_level 的流程 ---\n"
  done
done

echo "######################################################################"
echo "###   所有细胞类型和噪声等级的处理已全部完成！                 ###"
echo "######################################################################"
