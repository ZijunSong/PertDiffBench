#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of all datasets to be processed
DATASETS=(
    'mix2'
    'mix3'
    'mix4'
    'mix5'
    'mix6'
    'mix7'
)

# Define common parameters
NUM_GENES="4000"
N_SAMPLES="100"
NUM_RUNS=3
CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    echo "######################################################################"
    echo "###   Starting pipeline for dataset: $dataset"
    echo "######################################################################"

    # Define paths for the current dataset
    train_data_path="data/fig1/task3_hvg_output/${dataset}_train_HVG_${NUM_GENES}.h5ad"
    # Note: The user's example for eval also uses the training data. This script follows that pattern.
    eval_data_path="data/fig1/task3_hvg_output/${dataset}_train_HVG_${NUM_GENES}.h5ad"
    checkpoint_dir="checkpoints/fig1/task3/${dataset}/mlp_ddpm_mlp"
    samples_dir="samples/fig1/task3/${dataset}/mlp_ddpm_mlp"

    # --- Step 1: Training ---
    echo -e "\n--- Step 1: Training model for $dataset ---"
    python scripts/baseline/train_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "$train_data_path" \
        --save-weight-dir "$checkpoint_dir" \
        --gene-nums "$NUM_GENES"

    # --- Step 2: Evaluation (run multiple times) ---
    echo -e "\n--- Step 2: Evaluating model for $dataset ($NUM_RUNS runs) ---"
    
    all_outputs=""
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running evaluation iteration $i/$NUM_RUNS for $dataset ---"
        output=$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
            --config "$CONFIG_FILE" \
            --data-path "$eval_data_path" \
            --ckpt "${checkpoint_dir}/model_epoch_1000.pth" \
            --out_h5ad "${samples_dir}/synthetic_ifn_${i}.h5ad" \
            --gene-nums "$NUM_GENES" \
            --umap_plot "${samples_dir}/umap_comparison_${i}.png" \
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