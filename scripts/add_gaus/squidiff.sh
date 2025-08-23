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

# 定义通用参数
GENE_SIZE="6998"
NUM_RUNS=3

# 外层循环：遍历每种细胞类型
for cell_type in "${CELL_TYPES[@]}"; do
    # 中层循环：遍历每个噪声等级
    for noise_level in "${NOISE_LEVELS[@]}"; do
        echo "######################################################################"
        echo "###  处理细胞类型: $cell_type | 噪声等级: $noise_level"
        echo "######################################################################"

        # 动态构建带噪声的训练数据文件路径
        train_data_file="data/add_gaussian_noise_output/task1_train_CD4T_exp_noise_std_${noise_level}.h5ad"

        # 检查带噪声的训练文件是否存在，如果不存在则跳过
        if [ ! -f "$train_data_file" ]; then
            echo "警告: 未找到训练数据文件 '$train_data_file'。将跳过此组合。"
            continue
        fi

        # --- 第 1 步: 为当前细胞类型和噪声等级训练模型 (运行一次) ---
        echo -e "\n--- 正在为 $cell_type (噪声: $noise_level) 训练模型 ---"
        
        # 动态定义检查点目录
        checkpoint_dir="checkpoints/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}"
        mkdir -p "$checkpoint_dir"

        # 运行训练脚本
        python src/Squidiff/train_squidiff.py \
            --logger_path "logs/squidiff/${cell_type}_train_HVG_${GENE_SIZE}_noise_${noise_level}" \
            --data_path "$train_data_file" \
            --resume_checkpoint "$checkpoint_dir" \
            --gene_size "$GENE_SIZE" \
            --output_dim "$GENE_SIZE" 2>&1 | tee "logs/train_${cell_type}_noise_${noise_level}.log"

        echo "--- 为 $cell_type (噪声: $noise_level) 的训练已完成。 ---"

        # 用于存储所有推断运行的累积输出的变量
        all_outputs=""

        # --- 第 2 步: 为当前细胞类型和噪声等级多次运行推断 ---
        echo -e "\n--- 正在为 $cell_type (噪声: $noise_level) 开始推断 ($NUM_RUNS 次运行) ---"
        for (( i=1; i<=NUM_RUNS; i++ )); do
            echo -e "\n--- 正在为 $cell_type (噪声: $noise_level) 运行第 $i/$NUM_RUNS 次推断迭代 ---"
            
            # 运行 sample_squidiff.py 脚本
            output=$(python src/Squidiff/sample_squidiff.py \
                --model_path "${checkpoint_dir}/model.pt" \
                --gene_size "$GENE_SIZE" \
                --output_dim "$GENE_SIZE" \
                --out_h5ad "samples/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}/synthetic_ifn_run_${i}.h5ad" \
                --n_samples 6 \
                --umap_plot "samples/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}/umap_comparison_${i}.png" \
                --data_path "data/add_gaussian_noise_output/task1_valid_CD4T_exp_noise_std_${noise_level}.h5ad" 2>&1) || true
            
            # 打印当前运行的输出
            echo "$output"
            
            # 将当前输出附加到累积变量，用换行符分隔
            all_outputs+="$output\n"
        done

        # --- 第 3 步: 使用 AWK 进行统计计算和打印 ---
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
