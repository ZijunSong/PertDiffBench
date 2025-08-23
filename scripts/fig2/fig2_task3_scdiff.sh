#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status and print ERROR.
trap "echo ERROR && exit 1" ERR

# --------------------
# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"
NUM_RUNS=3
# --------------------

# Set HOMEDIR to the project root, assuming the script is in a subdirectory.
HOMEDIR=$(dirname $(dirname $(realpath $0)))/..
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# Define an array of the target species for the evaluation phase
TARGET_SPECIES=(
    'pig'
    'rabbit'
    'rat'
)

# Loop through each target species
for species in "${TARGET_SPECIES[@]}"; do
    # Construct the specific names for this run
    dataset_name="${species}_control_ifn"
    train_fname="mouse_control_ifn.h5ad"
    test_fname="${species}_control_ifn.h5ad"

    echo "######################################################################"
    echo "###   Starting pipeline for target species: $species"
    echo "######################################################################"

    # Construct the data settings string for the current dataset
    # Note: The train dataset name is based on the target species, but the train file name is always the mouse data.
    data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
    data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${test_fname}"

    # Variable to store the cumulative output of all runs
    all_outputs=""

    # Inner loop to run the script multiple times
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running iteration $i/$NUM_RUNS for $species ---"
        
        # This command runs the entire training and evaluation pipeline.
        # We run it multiple times to get multiple sets of evaluation metrics.
        output=$(python src/scDiff/main.py \
            --custom_data_path data/fig2/task3_cross_species \
            --base configs/scdiff/eval_perturbation.yaml \
            --name "${NAME}" \
            --logdir "${LOGDIR}" \
            --postfix "perturbation_${NAME}" \
            ${OFFLINE_SETTINGS} \
            ${data_settings} 2>&1) || true
        
        echo "$output"
        all_outputs+="$output\n"
    done

    # --- Statistical Calculation using AWK ---
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
