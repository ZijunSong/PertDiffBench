#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status and print ERROR.
trap "echo ERROR && exit 1" ERR

# --------------------
# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"
SEED=123 # Hardcoded seed for a single run
# --------------------

# Set HOMEDIR to the project root, assuming the script is in a subdirectory.
HOMEDIR=$(dirname $(dirname $(realpath $0)))/..
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# Construct the specific names for this run based on the hardcoded seed
dataset_name="seed${SEED}_control_train"
train_fname="seed${SEED}_control_train.h5ad"
test_fname="seed${SEED}_control_test.h5ad"

echo "######################################################################"
echo "###   Starting pipeline for dataset: $dataset_name"
echo "######################################################################"

# Construct the data settings string for the current dataset
data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${test_fname}"

# This command runs the entire training and evaluation pipeline.
# The output is piped to tee to show it on the console in real-time, 
# and then piped to awk for parsing and summarizing the results.
python src/scDiff/main.py \
    --custom_data_path data/fig2/task1_unseen_pert \
    --base configs/scdiff/eval_perturbation.yaml \
    --name "${NAME}" \
    --logdir "${LOGDIR}" \
    --postfix "perturbation_${NAME}" \
    ${OFFLINE_SETTINGS} \
    ${data_settings} 2>&1 | tee /dev/tty | awk -v dataset="$dataset_name" '
    # AWK script starts: capture all metrics from the script output
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

    # Reusable function to calculate and print statistics.
    # For a single run, mean is the value itself and std_dev is 0.
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
            
            # Print formatted stats. For a single run, it will show "value ± 0.0000"
            printf "%-40s: %.4f ± %.4f\n", name, mean, std_dev;
        } else {
            printf "%-40s: N/A (No data collected)\n", name;
        }
    }

    END {
        print "\n==================================================================";
        printf " Final statistics for %s (1 run)\n", dataset;
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

echo "######################################################################"
echo "###   Pipeline processing is complete!                             ###"
echo "######################################################################"
