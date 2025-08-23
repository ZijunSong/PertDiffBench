#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of all cell types to be processed
CELL_TYPES=(
    'B'
    'CD4T'
    'CD8T'
    'CD14+Mono'
    'Dendritic'
    'FCGR3A+Mono'
    'NK'
)

# Define common parameters
NUM_GENES="1000"
NUM_RUNS=3
CONFIG_FILE="configs/baselines/scrna_ddpm_scrna.yaml"

# Loop through each cell type
for cell_type in "${CELL_TYPES[@]}"; do
    echo "######################################################################"
    echo "###   Starting pipeline for cell type: $cell_type"
    echo "######################################################################"

    # Determine the correct training data file (handling the anomaly for CD4T)
    train_data_split="train"
    if [ "$cell_type" == "CD4T" ]; then
        # The user's original code used 'valid' for CD4T training, this preserves that.
        # Change to 'train' if this was a typo.
        train_data_split="valid"
    fi
    train_data_path="data/fig1/task1_hvg_output/${cell_type}_${train_data_split}_HVG_${NUM_GENES}.h5ad"
    valid_data_path="data/fig1/task1_hvg_output/${cell_type}_valid_HVG_${NUM_GENES}.h5ad"

    # --- Step 1: Training ---
    echo -e "\n--- Step 1: Training model for $cell_type ---"
    python scripts/baseline/train_scrna_ddpm_scrna.py \
        --config "$CONFIG_FILE" \
        --data-path "$train_data_path" \
        --save-weight-dir "checkpoints/fig1/task1/${cell_type}/scrna_ddpm_scrna" \
        --gene-nums "$NUM_GENES"

    # --- Step 2: Evaluation (run multiple times) ---
    echo -e "\n--- Step 2: Evaluating model for $cell_type ($NUM_RUNS runs) ---"
    
    all_outputs=""
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running evaluation iteration $i/$NUM_RUNS for $cell_type ---"
        output=$(python scripts/baseline/eval_scrna_ddpm_scrna.py \
            --config "$CONFIG_FILE" \
            --data-path "$valid_data_path" \
            --ckpt "checkpoints/fig1/task1/${cell_type}/scrna_ddpm_scrna/scrna_ddpm_epoch1000.pt" \
            --out_h5ad "samples/fig1/task1/${cell_type}/scrna_ddpm_scrna/synthetic_ifn_${i}.h5ad" \
            --gene-nums "$NUM_GENES" \
            --umap_plot "samples/fig1/task1/${cell_type}/scrna_ddpm_scrna/umap_comparison_${i}.png" \
            --n_samples 6 2>&1) || true
        
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
