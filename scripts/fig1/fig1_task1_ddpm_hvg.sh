#!/bin/bash

# 如果任何命令失败，立即退出脚本
set -e

# --- 配置区域 ---

# 定义要处理的基因数量列表
GENE_NUMS_LIST=(6998 6000 5000 4000 3000 2000 1000)

# 定义每次训练后要运行的评估次数
NUM_EVAL_RUNS=3

# 定义共享的配置文件路径
CONFIG_FILE="configs/baselines/scrna_ddpm_scrna.yaml"

# --- 脚本主逻辑 ---

# 遍历基因数量列表中的每一个值
for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   开始处理流程: 基因数量 = $gene_num"
    echo "######################################################################"

    # --- 动态路径设置 ---
    # 根据基因数量确定数据和输出目录的路径
    
    # 特殊处理基因数量为 6998 的情况，因为它的路径格式不同
    if [ "$gene_num" -eq 6998 ]; then
        train_data_path="data/fig1/task1/task1_train_CD4T_exp.h5ad"
        valid_data_path="data/fig1/task1/task1_valid_CD4T_exp.h5ad"
    else
        train_data_path="data/fig1/task1_hvg_output/CD4T_train_HVG_${gene_num}.h5ad"
        # 假设验证数据也遵循相似的命名规则
        valid_data_path="data/fig1/task1_hvg_output/CD4T_valid_HVG_${gene_num}.h5ad"
    fi

    save_dir_base="checkpoints/fig1/task1/scrna_ddpm_scrna_${gene_num}"
    sample_dir_base="samples/fig1/task1/scrna_ddpm_scrna_${gene_num}"
    checkpoint_file="${save_dir_base}/scrna_ddpm_epoch1000.pt"

    # --- 步骤 1: 模型训练 ---
    echo -e "\n--- 步骤 1: 训练模型 (基因数量: $gene_num) ---"
    python scripts/baseline/train_scrna_ddpm_scrna.py \
        --config "$CONFIG_FILE" \
        --data-path "$train_data_path" \
        --save-weight-dir "$save_dir_base" \
        --gene-nums "$gene_num"

    # --- 步骤 2: 多次模型评估 ---
    echo -e "\n--- 步骤 2: 评估模型 (基因数量: $gene_num, 运行 $NUM_EVAL_RUNS 次) ---"
    
    # 用于存储所有评估运行的输出
    all_outputs=""

    for (( i=1; i<=NUM_EVAL_RUNS; i++ )); do
        echo -e "\n--- 正在进行第 $i/$NUM_EVAL_RUNS 次评估 ---"
        
        # 捕获评估脚本的输出。`|| true`确保即使脚本返回错误码，循环也能继续
        output=$(python scripts/baseline/eval_scrna_ddpm_scrna.py \
            --config "$CONFIG_FILE" \
            --data-path "$valid_data_path" \
            --ckpt "$checkpoint_file" \
            --out_h5ad "${sample_dir_base}/synthetic_ifn_run${i}.h5ad" \
            --gene-nums "$gene_num" 2>&1) || true
        
        echo "$output"
        # 将本次运行的输出追加到总输出中
        all_outputs+="$output\n"
    done

    # --- 步骤 3: 使用 AWK 进行统计计算 ---
    echo -e "\n"
    echo "$all_outputs" | awk -v dataset="$cell_type" -v num_runs="$NUM_RUNS" '
        # AWK script starts: capture all metrics from the new eval script output
        /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
        /Mean Absolute Error \(MAE\):/ { mae[c_mae++] = $NF }
        /Differential Expression Score \(DES\):/ { des[c_des++] = $NF }
        /E-Distance:/ { edist[c_edist++] = $NF }
        /Maximum Mean Discrepancy \(MMD\):/ { mmd[c_mmd++] = $NF }
        /R-squared \(R2\):/ { r2[c_r2++] = $NF }
        /Pearson \(all genes\):/ { pearson_all[c_pearson_all++] = $NF }
        /Pearson Delta \(all genes\):/ { pearson_delta_all[c_pearson_delta_all++] = $NF }
        /Pearson Delta \(top 20 DE genes\):/ { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
        /Pearson Delta \(top 50 DE genes\):/ { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
        /Pearson Delta \(top 100 DE genes\):/ { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

        # Reusable function to calculate and print mean/std_dev
        function print_stat(name, data, count) {
            if (count > 0) {
                sum = 0;
                for (i = 0; i < count; i++) {
                    sum += data[i];
                }
                mean = sum / count;
                
                sum_sq_diff = 0;
                for (i = 0; i < count; i++) {
                    sum_sq_diff += (data[i] - mean)^2;
                }
                std_dev = (count > 1) ? sqrt(sum_sq_diff / (count - 1)) : 0;
                
                printf "%-40s: %.4f ± %.4f\n", name, mean, std_dev;
            } else {
                printf "%-40s: N/A (No data collected)\n", name;
            }
        }

        END {
            print "==================================================================";
            printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
            print "==================================================================";
            
            print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
            print_stat("Mean Absolute Error (MAE)", mae, c_mae);
            print_stat("Differential Expression Score (DES)", des, c_des);
            print "----------------------------------------";
            print_stat("E-Distance", edist, c_edist);
            print_stat("Maximum Mean Discrepancy (MMD)", mmd, c_mmd);
            print_stat("R-squared (R2)", r2, c_r2);
            print "----------------------------------------";
            print_stat("Pearson (all genes)", pearson_all, c_pearson_all);
            print_stat("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all);
            print_stat("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20);
            print_stat("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50);
            print_stat("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100);

            print "==================================================================\n";
        }
    '
    
    echo -e "\n--- Finished pipeline for cell type: $cell_type ---\n"
done

echo "######################################################################"
echo "###   All cell type processing is complete!                        ###"
echo "######################################################################"
