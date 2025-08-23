#!/bin/bash

# 如果任何命令以非零状态退出，则立即退出脚本。
set -e

# 定义要评估的所有细胞类型的数组
CELL_TYPES=(
    # 'B'
    'CD4T'
    # 'CD8T'
    # 'CD14+Mono'
    # 'Dendritic'
    # 'FCGR3A+Mono'
    # 'NK'
)

# 新增：定义要评估的所有噪声等级（标准差）的数组
NOISE_LEVELS=(0.1 0.25 0.5 1.0 1.5)

# 定义重复运行的次数
NUM_RUNS=3

# 外层循环：遍历每种细胞类型
for cell_type in "${CELL_TYPES[@]}"; do
    # 中层循环：遍历每个噪声等级
    for noise_level in "${NOISE_LEVELS[@]}"; do
        echo "######################################################################"
        echo "###  处理细胞类型: $cell_type | 噪声等级: $noise_level ($NUM_RUNS 次运行)"
        echo "######################################################################"

        # 用于存储所有运行的累积输出的变量
        all_outputs=""

        # 内层循环：多次运行评估脚本
        for (( i=1; i<=NUM_RUNS; i++ )); do
            echo -e "\n--- 正在为 $cell_type (噪声: $noise_level) 运行第 $i/$NUM_RUNS 次迭代 ---"
            
            # 动态构建带噪声的训练数据文件路径
            # 假设您的噪声数据文件遵循 "原始文件名_noise_std_噪声值.h5ad" 的格式
            train_data_file="data/add_gaussian_noise_output/task1_train_${cell_type}_exp_noise_std_${noise_level}.h5ad"

            # 检查文件是否存在，如果不存在则跳过
            if [ ! -f "$train_data_file" ]; then
                echo "警告: 未找到训练数据文件 '$train_data_file'。将跳过此组合。"
                continue 2 # 跳出内层循环，继续下一个噪声等级
            fi

            # 运行原始的 Python 评估脚本，并将其输出（stdout 和 stderr）附加到 all_outputs 变量
            # 使用 '|| true' 防止 'set -e' 在 python 脚本向 stderr 输出警告时退出脚本
            output=$(python scripts/scGen_eval.py \
                --train_data_path "$train_data_file" \
                --test_data_path "data/fig1/task1/task1_valid_${cell_type}_exp.h5ad" \
                --model_save_path "checkpoints/scgen/${cell_type}_${noise_level}_6998" \
                --out_h5ad "samples/fig1/task1/scgen/${cell_type}_${noise_level}_6998_pred_${i}.h5ad" \
                --umap_plot "samples/fig1/task1/scgen/${cell_type}_${noise_level}_umap_comparison_${i}.png" \
                --n_samples 6 \
                --celltype_to_predict "$cell_type" 2>&1) || true
            
            # 打印当前运行的输出
            echo "$output"
            
            # 将当前输出附加到累积变量，用换行符分隔
            all_outputs+="$output\n"
        done

        # --- 使用 AWK 进行统计计算和打印 ---
        # 将所有运行的累积输出通过管道传递给 awk
        # awk 脚本解析所有指标，计算均值和标准差，并按指定格式打印
        echo -e "\n"
        echo "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" '
            # AWK 脚本开始：从新的评估脚本输出中捕获所有指标
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

            # 可重用函数，用于计算和打印均值/标准差
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
                    printf "%-40s: N/A (未收集到数据)\n", name;
                }
            }

            END {
                print "==================================================================";
                printf " %s (噪声: %s) 的最终统计结果 (%d 次运行)\n", dataset, noise, num_runs;
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
        
        echo -e "\n--- 完成细胞类型: $cell_type | 噪声等级: $noise_level 的流程 ---\n"
    done # 噪声等级循环结束
done # 细胞类型循环结束

echo "######################################################################"
echo "###   所有细胞类型和噪声等级的处理已全部完成！                 ###"
echo "######################################################################"

