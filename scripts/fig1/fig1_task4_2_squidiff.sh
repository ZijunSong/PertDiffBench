#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array to hold the dataset prefixes.
# The script will construct the full paths based on these prefixes.
declare -A DATASETS
DATASETS=(
    ["ACTA2"]="5737"
    ["B2M"]="5737"
)

# Define the number of replicate sampling runs
NUM_RUNS=3

# First, change to the directory where the python scripts are located.
echo "Changing directory to src/Squidiff..."
cd src/Squidiff

# Loop through each dataset prefix (which are the keys of the associative array).
for prefix in "${!DATASETS[@]}"; do
    # Get the corresponding gene size for the current dataset.
    gene_size=${DATASETS[$prefix]}
    
    # Construct the specific names for this run
    model_name="task4_${prefix}_control_to_ifn"
    train_data_name="task4_${prefix}_control_to_ifn"
    test_data_name="task4_${prefix}_control_to_coculture"

    echo "######################################################################"
    echo "###   Starting to process model: $model_name"
    echo "###   Gene Size: $gene_size"
    echo "######################################################################"

    # --- Step 1: Training (run once per dataset) ---
    echo -e "\n--- Running training for $model_name ---"
    python train_squidiff.py \
        --logger_path "../../logs/squidiff/task4/${model_name}" \
        --data_path "../../data/fig1/task4/${train_data_name}.h5ad" \
        --resume_checkpoint "../../checkpoints/fig1/task4/${model_name}" \
        --gene_size "$gene_size" \
        --output_dim "$gene_size"

    # --- Step 2: Sampling and Evaluation (run multiple times) ---
    # Variable to store the cumulative output of all sampling runs
    all_outputs=""

    # Inner loop to run the sampling script multiple times
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running sampling iteration $i/$NUM_RUNS for $model_name ---"
        
        # Run the sampling script and capture its output
        output=$(python sample_squidiff.py \
            --model_path "../../checkpoints/fig1/task4/${model_name}/model.pt" \
            --gene_size "$gene_size" \
            --output_dim "$gene_size" \
            --out_h5ad "samples/fig1/task4_2/${cell_type}/squidiff_1000/synthetic_ifn_run_${i}.h5ad" \
            --n_samples 100 \
            --umap_plot "samples/fig1/task4_2/${cell_type}/squidiff_1000/umap_comparison_${i}.png" \
            --data_path "../../data/fig1/task4/${test_data_name}.h5ad" 2>&1) || true

        # Print the output of the current run
        echo "$output"
        
        # Append the current output to the cumulative variable
        all_outputs+="$output\n"
    done

    # --- Step 3: Statistical Calculation using AWK ---
    # Pipe the cumulative output of all sampling runs to awk
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
