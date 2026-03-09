#!/bin/bash
# 三次（训练+测评），聚合评测结果→写入 CSV

# 如果任何命令以非零状态退出，则立即退出脚本并打印错误。
trap "echo ERROR && exit 1" ERR
set -e

# --------------------
# 配置（仅路径相关最小改动）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=${NAME:-v7.5}
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiff}"          # 写到 CSV 里的方法名
METHOD_DIR="${METHOD_DIR:-scdiff}"            # 用在 samples 目录最后一级
# --------------------

# 数据根目录（统一入口）
DATA_ROOT="data/add_poisson_technoise_output"

# 将 HOMEDIR 设置为项目根目录，假设脚本位于子目录中。
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "$HOMEDIR"
echo "当前工作目录: $(pwd)"

# 日志根目录（按组合分别记录）
RUNLOG_ROOT="${LOGDIR}/perturbation_${NAME}"
mkdir -p "${RUNLOG_ROOT}"

# 定义要处理的所有细胞类型的数组
CELL_TYPES=(
  'CD4T'
)

# 要评估的噪声等级
NOISE_LEVELS=(
  '0.25'
  '0.5'
  '1.0'
  '2.0'
  '4.0'
)

# 外层循环：遍历每种细胞类型
for cell_type in "${CELL_TYPES[@]}"; do
  # 中层循环：遍历每个噪声等级
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###  处理细胞类型: $cell_type | 噪声等级: $noise_level"
    echo "######################################################################"

    # 动态构建带噪声的训练/验证数据文件名（路径统一到 DATA_ROOT）
    train_fname="task1_train_${cell_type}_exp_poisson_depth_${noise_level}.h5ad"
    valid_fname="task1_valid_${cell_type}_exp_poisson_depth_${noise_level}.h5ad"

    # 检查训练文件是否存在
    if [ ! -f "${DATA_ROOT}/${train_fname}" ]; then
      echo "警告: 未找到训练数据文件 '${DATA_ROOT}/${train_fname}'。跳过此组合。"
      continue
    fi

    # 数据设置字符串（仅路径名一致性）
    dataset_name="fig1_task1_${cell_type}_noise_${noise_level}"
    data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
    data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${valid_fname}"

    # 组合级别日志与输出目录
    COMBO_TAG="${cell_type}_noise_${noise_level}"
    DATASET_LOG="${RUNLOG_ROOT}/${COMBO_TAG}.log"
    output_suffix="${cell_type}_noise_${noise_level}"

    # 注意：samples 路径的最后一级目录名用 METHOD_DIR 控制
    OUTDIR_BASE="samples/poisson_technoise/${output_suffix}/${METHOD_DIR}"
    mkdir -p "${OUTDIR_BASE}"

    echo -e "\n==== $(date '+%F %T') | Begin ${COMBO_TAG} ====\n" | tee -a "${DATASET_LOG}"

    # 收集多次评测输出
    all_outputs=""

    # 内层循环：多次运行脚本（每次调用 main.py = 一次训练+评测）
    for (( i=1; i<=NUM_RUNS; i++ )); do
      echo -e "\n======================"                 | tee -a "${DATASET_LOG}"
      echo -e " Run ${i}/${NUM_RUNS} : ${COMBO_TAG}"    | tee -a "${DATASET_LOG}"
      echo -e "======================"                 | tee -a "${DATASET_LOG}"

      # 仅修改 --custom_data_path 指向 DATA_ROOT；其余参数保持你的调用习惯
      output=$(python src/scDiff/main.py \
        --custom_data_path "${DATA_ROOT}" \
        --base configs/scdiff/eval_perturbation.yaml \
        --name "${NAME}_${COMBO_TAG}_run${i}" \
        --logdir "${LOGDIR}" \
        --postfix "perturbation_${NAME}" \
        ${OFFLINE_SETTINGS} \
        ${data_settings} 2>&1) || true

      # 打印并写入日志
      echo "${output}" | tee -a "${DATASET_LOG}"

      # 累积到统计文本（保留换行）
      all_outputs+="${output}"$'\n'
    done

    # ==== 统计到控制台 + 写入 CSV ====
    CSV_FILE="${OUTDIR_BASE}/metrics_${METHOD_DIR}_${COMBO_TAG}.csv"
    mkdir -p "$(dirname "${CSV_FILE}")"

    echo -e "\n" | tee -a "${DATASET_LOG}"
    echo -e "${all_outputs}" | awk \
      -v method="${METHOD_NAME}" \
      -v num_runs="${NUM_RUNS}" \
      -v csv_path="${CSV_FILE}" \
      -v combo="${COMBO_TAG}" \
      -v ds="${cell_type}" \
      -v nz="${noise_level}" '
      BEGIN{
        c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=c_pearson_all=c_pearson_delta_all=c_pearson_delta_de20=c_pearson_delta_de50=c_pearson_delta_de100=0
      }
      function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

      # -------- 捕获 11 个指标（从日志中抽取数值）--------
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++]                   = to_num($NF); next }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++]                   = to_num($NF); next }
      /Differential Expression Score \(DES\):/    { des[c_des++]                   = to_num($NF); next }
      /E-Distance:/                               { edist[c_edist++]               = to_num($NF); next }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++]                   = to_num($NF); next }
      /R-squared \(R2\):/                         { r2[c_r2++]                     = to_num($NF); next }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++]   = to_num($NF); next }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = to_num($NF); next }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = to_num($NF); next }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = to_num($NF); next }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = to_num($NF); next }

      function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n : 0 }
      function std(a,n,  mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }

      function mean_std(idx,  n,mu,sd){
        if(idx==1){ n=c_pds;                  mu=mean(pds,n);                  sd=std(pds,n) }
        else if(idx==2){ n=c_mae;            mu=mean(mae,n);                  sd=std(mae,n) }
        else if(idx==3){ n=c_des;            mu=mean(des,n);                  sd=std(des,n) }
        else if(idx==4){ n=c_edist;          mu=mean(edist,n);                sd=std(edist,n) }
        else if(idx==5){ n=c_mmd;            mu=mean(mmd,n);                  sd=std(mmd,n) }
        else if(idx==6){ n=c_r2;             mu=mean(r2,n);                   sd=std(r2,n) }
        else if(idx==7){ n=c_pearson_all;    mu=mean(pearson_all,n);          sd=std(pearson_all,n) }
        else if(idx==8){ n=c_pearson_delta_all;   mu=mean(pearson_delta_all,n);   sd=std(pearson_delta_all,n) }
        else if(idx==9){ n=c_pearson_delta_de20;  mu=mean(pearson_delta_de20,n);  sd=std(pearson_delta_de20,n) }
        else if(idx==10){n=c_pearson_delta_de50;  mu=mean(pearson_delta_de50,n);  sd=std(pearson_delta_de50,n) }
        else if(idx==11){n=c_pearson_delta_de100; mu=mean(pearson_delta_de100,n); sd=std(pearson_delta_de100,n) }
        return sprintf("%.6f±%.6f", mu, sd)
      }

      function val(idx, j, v){
        if      (idx==1)  v=pds[j];
        else if (idx==2)  v=mae[j];
        else if (idx==3)  v=des[j];
        else if (idx==4)  v=edist[j];
        else if (idx==5)  v=mmd[j];
        else if (idx==6)  v=r2[j];
        else if (idx==7)  v=pearson_all[j];
        else if (idx==8)  v=pearson_delta_all[j];
        else if (idx==9)  v=pearson_delta_de20[j];
        else if (idx==10) v=pearson_delta_de50[j];
        else if (idx==11) v=pearson_delta_de100[j];
        return (v=="") ? 0 : v;
      }

      function print_stat(name, data, count,   i,s,mu,ss,std_val){
        if (count>0){
          for(i=0;i<count;i++) s+=data[i];
          mu=s/count;
          for(i=0;i<count;i++) ss+=(data[i]-mu)*(data[i]-mu);
          std_val=(count>1)?sqrt(ss/(count-1)):0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std_val;
        } else {
          printf "%-40s: N/A (未收集到数据)\n", name;
        }
      }

      END {
        print "==================================================================";
        printf " 最终统计：%s（%d 次运行）\n", combo, num_runs;
        print "==================================================================";

        print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
        print_stat("Mean Absolute Error (MAE)",         mae, c_mae);
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

        # ---------- CSV：Dataset, Noise, Method + mean±std + 每次运行 ----------
        metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES";
        metric_names[4]="E-Distance"; metric_names[5]="MMD"; metric_names[6]="R2";
        metric_names[7]="Pearson (all genes)";
        metric_names[8]="Pearson Delta (all genes)";
        metric_names[9]="Pearson Delta (top 20 DE genes)";
        metric_names[10]="Pearson Delta (top 50 DE genes)";
        metric_names[11]="Pearson Delta (top 100 DE genes)";

        header="Dataset,Noise,Method";
        for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
        for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

        row=ds "," nz "," method;
        for(i=1;i<=11;i++){
          ms=mean_std(i); split(ms, parts, "|");
          row=row sprintf(",%.6f±%.6f", parts[1], parts[2]);
        }
        for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

        print header > csv_path;
        print row    >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    ' | tee -a "${DATASET_LOG}"

    echo -e "\n--- 完成组合: ${COMBO_TAG} ---\n" | tee -a "${DATASET_LOG}"
  done
done

echo "######################################################################"
echo "###   所有细胞类型和噪声等级的处理已全部完成！                 ###"
echo "######################################################################"
