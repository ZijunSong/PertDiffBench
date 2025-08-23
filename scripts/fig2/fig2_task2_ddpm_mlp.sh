#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of the target cell types for the evaluation phase
TARGET_CELL_TYPES=(
    "B"
    "NK"
)

# Define common parameters
NUM_GENES="6998"
N_SAMPLES="54"
NUM_RUNS=3
CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"

# --- Step 1: Training (run once for the entire script) ---
echo "######################################################################"
echo "###   Step 1: Training model on pretrain_CD4T data"
echo "######################################################################"
python scripts/baseline/train_mlp_ddpm_mlp.py \
    --config "$CONFIG_FILE" \
    --data-path "data/fig1/task1/task1_train_CD4T_exp.h5ad" \
    --save-weight-dir "checkpoints/fig2/task2/pretrain_CD4T/mlp_ddpm_mlp" \
    --gene-nums "$NUM_GENES"

# --- Step 2: Loop through targets for Evaluation ---
for cell_type in "${TARGET_CELL_TYPES[@]}"; do
    
    descriptive_name="Evaluate_on_${cell_type}"

    echo -e "\n######################################################################"
    echo "###   Step 2: Evaluating model on target: $cell_type ($NUM_RUNS runs)"
    echo "######################################################################"
    
    all_outputs=""
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running evaluation iteration $i/$NUM_RUNS for $cell_type ---"
        output=$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
            --config "$CONFIG_FILE" \
            --train-data-path "data/fig1/task1/task1_train_CD4T_exp.h5ad" \
            --data-path "data/fig1/task1/task1_valid_${cell_type}_exp.h5ad" \
            --ckpt "checkpoints/fig2/task2/pretrain_CD4T/mlp_ddpm_mlp/model_epoch_1000.pth" \
            --out_h5ad "samples/fig2/task2_unseen_celltype/pretrain_CD4T_${cell_type}/mlp_ddpm_mlp/synthetic_ifn_${i}.h5ad" \
            --gene-nums "$NUM_GENES" \
            --umap_plot "samples/fig2/task2_unseen_celltype/pretrain_CD4T_${cell_type}/mlp_ddpm_mlp/umap_comparison_${i}.svg" \
            --n_samples "$N_SAMPLES" 2>&1) || true
        
        echo "$output"
        all_outputs+="$output\n"
    done

    # --- Step 3: Statistical Calculation using AWK ---
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