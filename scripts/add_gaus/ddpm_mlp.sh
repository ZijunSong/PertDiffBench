#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# ==============================================================================
# Script Configuration
# ==============================================================================

# Define an array of all cell types to be processed.
CELL_TYPES=(
    # 'B'
    'CD4T'
    # 'CD8T'
    # 'CD14+Mono'
    # 'Dendritic'
    # 'FCGR3A+Mono'
    # 'NK'
)

# Define the Gaussian noise standard deviation levels to iterate over.
NOISE_LEVELS=(
    '0.1'
    '0.25'
    '0.5'
    '1.0'
    '1.5'
)

# Define common parameters
NUM_GENES="6998"
NUM_RUNS=3
CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"
BASE_DATA_DIR="data/add_gaussian_noise_output" # Base directory for the new dataset

# ==============================================================================
# Main Processing Loop
# ==============================================================================

# Loop through each cell type
for cell_type in "${CELL_TYPES[@]}"; do
    # Loop through each noise level
    for noise_level in "${NOISE_LEVELS[@]}"; do
        echo "######################################################################"
        echo "###   Pipeline starting for: Cell Type = $cell_type | Noise Std = $noise_level"
        echo "######################################################################"

        # --- Define Dynamic File Paths ---
        # Construct training and validation data paths based on cell type and noise level
        train_data_path="${BASE_DATA_DIR}/task1_train_${cell_type}_exp_noise_std_${noise_level}.h5ad"
        valid_data_path="${BASE_DATA_DIR}/task1_valid_${cell_type}_exp_noise_std_${noise_level}.h5ad"
        
        # Create unique directory paths for checkpoints and output samples to avoid overwrites
        output_suffix="${cell_type}_noise_${noise_level}"
        save_weight_dir="checkpoints/gaussian_noise_mlp/${output_suffix}/mlp_ddpm_mlp_${NUM_GENES}"
        samples_dir="samples/gaussian_noise_mlp/${output_suffix}/mlp_ddpm_mlp_${NUM_GENES}"
        checkpoint_file="${save_weight_dir}/model_epoch_1000.pth"

        # --- Step 1: Training ---
        echo -e "\n--- Step 1: Training model for $cell_type with noise $noise_level ---"
        python scripts/baseline/train_mlp_ddpm_mlp.py \
            --config "$CONFIG_FILE" \
            --data-path "$train_data_path" \
            --save-weight-dir "$save_weight_dir" \
            --gene-nums "$NUM_GENES"

        # --- Step 2: Evaluation (run multiple times) ---
        echo -e "\n--- Step 2: Evaluating model for $cell_type with noise $noise_level ($NUM_RUNS runs) ---"
        
        all_outputs=""
        for (( i=1; i<=NUM_RUNS; i++ )); do
            echo -e "\n--- Running evaluation iteration $i/$NUM_RUNS for $cell_type (Noise: $noise_level) ---"
            
            # Run the evaluation script, capturing both stdout and stderr
            output=$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
                --config "$CONFIG_FILE" \
                --data-path "$valid_data_path" \
                --ckpt "$checkpoint_file" \
                --out_h5ad "${samples_dir}/synthetic_ifn_${i}.h5ad" \
                --umap_plot "${samples_dir}/umap_comparison_${i}.png" \
                --gene-nums "$NUM_GENES" \
                --n_samples 6 2>&1) || true # '|| true' ensures the bash script does not exit due to 'set -e' if the python script fails
            
            echo "$output"
            all_outputs+="$output\n"
        done

        # --- Step 3: Statistical Calculation using AWK ---
        echo -e "\n--- Step 3: Calculating statistics for $cell_type with noise $noise_level ---"
        echo "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" '
            # AWK script starts: capture all metrics from the eval script output
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
                printf " Final statistics for %s (Noise Std: %s) (%d runs)\n", dataset, noise, num_runs;
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
        
        echo -e "\n--- Finished pipeline for: Cell Type = $cell_type | Noise Std = $noise_level ---\n"
    done # End of noise_level loop
done # End of cell_type loop

echo "######################################################################"
echo "###   All processing is complete!                                  ###"
echo "######################################################################"
