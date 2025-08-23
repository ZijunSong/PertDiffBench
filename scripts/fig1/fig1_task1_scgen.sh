#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of all cell types to be evaluated
CELL_TYPES=(
    'B'
    'CD4T'
    'CD8T'
    'CD14+Mono'
    'Dendritic'
    'FCGR3A+Mono'
    'NK'
)

# Define the number of replicate runs
NUM_RUNS=3

# Loop through each cell type
for cell_type in "${CELL_TYPES[@]}"; do
    echo "######################################################################"
    echo "###   Starting to process cell type: $cell_type ($NUM_RUNS runs total)"
    echo "######################################################################"

    # Variable to store the cumulative output of all runs
    all_outputs=""

    # Inner loop to run the evaluation script multiple times
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running iteration $i/$NUM_RUNS for $cell_type ---"
        
        # Run the original Python evaluation script and append its output (stdout and stderr) to the all_outputs variable
        # Use '|| true' to prevent 'set -e' from exiting the script if the python script outputs warnings to stderr
        output=$(python scripts/scGen_eval.py \
            --train_data_path "data/fig1/task1/task1_train_${cell_type}_exp.h5ad" \
            --test_data_path "data/fig1/task1/task1_valid_${cell_type}_exp.h5ad" \
            --model_save_path "checkpoints/scgen/${cell_type}_6998" \
            --out_h5ad "samples/fig1/task1/scgen/${cell_type}_6998_pred_${i}.h5ad" \
            --umap_plot "samples/fig1/task1/scgen/${cell_type}_umap_comparison_${i}.png" \
            --n_samples 6 \
            --celltype_to_predict "$cell_type" 2>&1) || true
        
        # Print the output of the current run
        echo "$output"
        
        # Append the current output to the cumulative variable, separated by a newline
        all_outputs+="$output\n"
    done

    # --- Use AWK for statistical calculation and printing ---
    # Pipe the cumulative output of all runs to awk
    # The awk script parses all metrics, calculates mean and standard deviation, and prints in the specified format
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
