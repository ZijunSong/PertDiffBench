#!/bin/bash

# 如果任何命令失败，立即退出脚本，并打印错误信息
trap 'echo "ERROR: A command failed. Exiting." >&2; exit 1' ERR

# --- 配置区域 ---

# 定义要处理的基因数量列表
GENE_NUMS_LIST=(6998 6000 5000 4000 3000 2000 1000)

# 定义为获取统计结果而需要重复运行评估的次数
NUM_EVAL_RUNS=3

# 定义共享的脚本参数
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"

# --- 脚本主逻辑 ---

# 遍历基因数量列表中的每一个值
for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   开始处理 scDiff 流程: 基因数量 = $gene_num"
    echo "######################################################################"

    # --- 动态路径和参数设置 ---
    data_settings=""
    if [ "$gene_num" -eq 6998 ]; then
        dataset_name="fig1_task1_CD4T"
        train_fname="task1_train_CD4T_exp.h5ad"
        valid_fname="task1_valid_CD4T_exp.h5ad"
        data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
        data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${valid_fname}"
        CUSTOM_DATA_PATH="data/fig1/task1"
    else
        dataset_name="fig1_task1_CD4T_${gene_num}"
        train_fname="CD4T_train_HVG_${gene_num}.h5ad"
        valid_fname="CD4T_valid_HVG_${gene_num}.h5ad"
        data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
        data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${valid_fname}"
        CUSTOM_DATA_PATH="data/fig1/task1_hvg_output"
    fi

    # --- 步骤 1: 多次运行训练与评估 ---
    echo -e "\n--- 步骤 1: 运行训练与评估 (基因数量: $gene_num, 共 $NUM_EVAL_RUNS 次) ---"
    all_outputs=""
    for (( i=1; i<=NUM_EVAL_RUNS; i++ )); do
        echo -e "\n--- 正在进行第 $i/$NUM_EVAL_RUNS 次运行 ---"
        
        # 捕获评估脚本的输出
        # 我们为每次运行添加一个独特的后缀，以防日志冲突
        run_postfix="perturbation_${NAME}_gene${gene_num}_run${i}"
        
        output=$(python src/scDiff/main.py \
            --custom_data_path "${CUSTOM_DATA_PATH}" \
            --base configs/scdiff/eval_perturbation.yaml \
            --name "${NAME}" \
            --logdir "${LOGDIR}" \
            --postfix "${run_postfix}" \
            ${OFFLINE_SETTINGS} \
            ${data_settings} 2>&1) || true
        
        echo "$output"
        all_outputs+="$output\n"
    done

    # --- 步骤 2: 使用 AWK 进行统计计算 ---
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
