#!/bin/bash

# 如果任何命令失败，立即退出脚本
set -e

# 定义要处理的基因数量列表
# GENE_NUMS_LIST=(6998 6000 5000)
# GENE_NUMS_LIST=(4000 3000)
GENE_NUMS_LIST=(2000 1000)

# 定义为获取统计结果而需要重复运行评估的次数
NUM_EVAL_RUNS=3

# --- 脚本主逻辑 ---

# 遍历基因数量列表中的每一个值
for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   开始处理 scDiffusion 流程: 基因数量 = $gene_num"
    echo "######################################################################"

    # --- 动态路径设置 ---
    # 根据基因数量确定数据和输出目录的路径
    if [ "$gene_num" -eq 6998 ]; then
        train_data_path="data/fig1/task1/task1_train_CD4T_exp.h5ad"
        valid_data_path="data/fig1/task1/task1_valid_CD4T_exp.h5ad"
    else
        train_data_path="data/fig1/task1_hvg_output/CD4T_train_HVG_${gene_num}.h5ad"
        valid_data_path="data/fig1/task1_hvg_output/CD4T_valid_HVG_${gene_num}.h5ad"
    fi

    # 定义检查点和输出目录
    vae_checkpoint_dir="checkpoints/scdiffusion/vae_checkpoint/task1/CD4T_${gene_num}"
    diffusion_checkpoint_dir="../../checkpoints/scdiffusion/diffusion_checkpoint/task1/CD4T_${gene_num}"
    classifier_checkpoint_dir="../../checkpoints/scdiffusion/classifier_checkpoint/2-classifier/task1/CD4T_${gene_num}"
    sample_output_dir="../../../samples/fig1/task1/CD4T_${gene_num}/scDiffusion"

    # 定义具体的检查点文件路径
    vae_model_file="${vae_checkpoint_dir}/model_seed=0_step=9999.pt"
    diffusion_model_file="${diffusion_checkpoint_dir}/my_diffusion/model010000.pt"
    classifier_model_file="${classifier_checkpoint_dir}/model009999.pt"

    # --- 步骤 1: 训练 VAE (Autoencoder) ---
    echo -e "\n--- 步骤 1: 训练 VAE (基因数量: $gene_num) ---"
    cd src/scDiffusion/VAE
    python VAE_train.py \
        --data_dir "../../../${train_data_path}" \
        --num_genes "$gene_num" \
        --state_dict "../../../checkpoints/scimilarity/model_v1.1" \
        --save_dir "../../../${vae_checkpoint_dir}"
    cd ..

    # --- 步骤 2: 训练 Diffusion Backbone ---
    echo -e "\n--- 步骤 2: 训练 Diffusion Backbone (基因数量: $gene_num) ---"
    python cell_train.py \
        --data_dir "../../${train_data_path}" \
        --vae_path "../../${vae_model_file}" \
        --save_dir "$diffusion_checkpoint_dir"

    # --- 步骤 3: 训练 Classifier ---
    echo -e "\n--- 步骤 3: 训练 Classifier (基因数量: $gene_num) ---"
    python classifier_train.py \
        --data_dir "../../${train_data_path}" \
        --vae_path "../../${vae_model_file}" \
        --model_path "$classifier_checkpoint_dir"

    # --- 步骤 4: 多次运行扰动预测与评估 ---
    echo -e "\n--- 步骤 4: 运行预测与评估 (基因数量: $gene_num, 共 $NUM_EVAL_RUNS 次) ---"
    all_outputs=""
    for (( i=1; i<=NUM_EVAL_RUNS; i++ )); do
        echo -e "\n--- 正在进行第 $i/$NUM_EVAL_RUNS 次评估 ---"
        
        # 捕获评估脚本的输出
        output=$(python classifier_sample.py \
            --num_samples 100 \
            --train-data-path "../../${train_data_path}" \
            --model_path "$diffusion_model_file" \
            --classifier_path "$classifier_model_file" \
            --ae_dir "../../${vae_model_file}" \
            --num_gene "$gene_num" \
            --sample_dir "${sample_output_dir}_run${i}" \
            --init_cell_path "../../${valid_data_path}" 2>&1) || true
        
        echo "$output"
        all_outputs+="$output\n"
    done
    cd ../..

    # --- 步骤 5: 使用 AWK 进行统计计算 ---
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
