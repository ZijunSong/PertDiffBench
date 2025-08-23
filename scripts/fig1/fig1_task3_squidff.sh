#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of mix types to be evaluated for Task 3
MIX_TYPES=(
    'mix2'
    'mix3'
    'mix4'
    'mix5'
    'mix6'
    'mix7'
)

# Define the number of replicate inference runs
NUM_RUNS=3

# --- Process Task 3 Mix Types ---
for mix_type in "${MIX_TYPES[@]}"; do
    echo "######################################################################"
    echo "###   Starting to process mix type (Task 3): $mix_type"
    echo "######################################################################"

    # --- Step 1: Train the model for the current mix type (run once) ---
    echo -e "\n--- Training model for $mix_type ---"
    # Ensure the checkpoint directory exists before training
    mkdir -p "checkpoints/fig1/task3/${mix_type}/squidiff_1000"

    # Run the training script for Task 3
    python src/Squidiff/train_squidiff.py \
        --logger_path logs/squidiff/${mix_type}_train_HVG_1000 \
        --data_path "data/fig1/task3_hvg_output/${mix_type}_train_HVG_1000.h5ad" \
        --resume_checkpoint "checkpoints/fig1/task3/${mix_type}/squidiff_1000" \
        --gene_size 1000 \
        --output_dim 1000 2>&1 | tee "logs/train_${mix_type}.log" # Log training output

    echo "--- Training for $mix_type complete. ---"

    # Variable to store the cumulative output of all inference runs
    all_outputs=""

    # --- Step 2: Run inference multiple times for the current mix type ---
    echo -e "\n--- Starting inference for $mix_type ($NUM_RUNS runs total) ---"
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running inference iteration $i/$NUM_RUNS for $mix_type ---"
        
        # Run the sample_squidiff.py script for Task 3
        output=$(python src/Squidiff/sample_squidiff.py \
            --model_path "checkpoints/fig1/task3/${mix_type}/squidiff_1000/model.pt" \
            --gene_size 1000 \
            --output_dim 1000 \
            --out_h5ad "samples/fig1/task3/${cell_type}/squidiff_1000/synthetic_ifn_run_${i}.h5ad" \
            --n_samples 100 \
            --umap_plot "samples/fig1/task3/${cell_type}/squidiff_1000/umap_comparison_${i}.png" \
            --data_path "data/fig1/task3_hvg_output/${mix_type}_test_HVG_1000.h5ad" 2>&1) || true
        
        # Print the output of the current run
        echo "$output"
        
        # Append the current output to the cumulative variable, separated by a newline
        all_outputs+="$output\n"
    done

    # --- Step 3: Use AWK for statistical calculation and printing ---
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
