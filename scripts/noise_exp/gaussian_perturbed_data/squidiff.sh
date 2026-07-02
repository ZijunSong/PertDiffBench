#!/bin/bash

# Exit immediately if any command fails.
set -e

source "scripts/lib/max_n_samples.sh"

# definemustevalallcelltypearray
CELL_TYPES=(
    # 'B'
    'CD4T'
    # 'CD8T'
    # 'CD14+Mono'
    # 'Dendritic'
    # 'FCGR3A+Mono'
    # 'NK'
)

# : definemustevalallnoise level ( )array
NOISE_LEVELS=(0.1 0.25 0.5 1.0 1.5)

# define usingargs
GENE_SIZE="6998"
NUM_RUNS=3

# outer loop: loop cell types
for cell_type in "${CELL_TYPES[@]}"; do
    # middle loop: loop noise levels
    for noise_level in "${NOISE_LEVELS[@]}"; do
        echo "######################################################################"
        echo "###  Processing: $cell_type | noise: $noise_level"
        echo "######################################################################"

        # build noisy train pathdatafilepath
        train_data_file="data/add_gaussian_noise_output/task1_train_CD4T_exp_noise_std_${noise_level}.h5ad"

        # check noisetrainfilewhetherexist, exist skip
        if [ ! -f "$train_data_file" ]; then
            echo " : Training data file not found '$train_data_file'.skip this combo."
            continue
        fi

        # --- 1 : ascurrentcelltype noise leveltrain ( ) ---
        echo -e "\n--- Running $cell_type (noise: $noise_level) train ---"
        
        # definecheck directory
        checkpoint_dir="checkpoints/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}"
        mkdir -p "$checkpoint_dir"

        # train 
        python src/Squidiff/train_squidiff.py \
            --logger_path "logs/squidiff/${cell_type}_train_HVG_${GENE_SIZE}_noise_${noise_level}" \
            --data_path "$train_data_file" \
            --resume_checkpoint "$checkpoint_dir" \
            --gene_size "$GENE_SIZE" \
            --output_dim "$GENE_SIZE" 2>&1 | tee "logs/train_${cell_type}_noise_${noise_level}.log"

        echo "--- as $cell_type (noise: $noise_level) train done. ---"

        # for all output amount
        all_outputs=""

        valid_h5="data/add_gaussian_noise_output/task1_valid_CD4T_exp_noise_std_${noise_level}.h5ad"
        N_SAMPLES="$(max_n_samples_paired "${valid_h5}")"

        # --- 2 : ascurrentcelltype noise level ---
        echo -e "\n--- Running $cell_type (noise: $noise_level) Start ($NUM_RUNS ) ---"
        for (( i=1; i<=NUM_RUNS; i++ )); do
          export RUN_SEED=$(($i-1))
            echo -e "\n--- Running $cell_type (noise: $noise_level) run $i/$NUM_RUNS  inference iterations ---"
            
            # sample_squidiff.py 
            output=$(python src/Squidiff/sample_squidiff.py \
                --model_path "${checkpoint_dir}/model.pt" \
                --gene_size "$GENE_SIZE" \
                --output_dim "$GENE_SIZE" \
                --out_h5ad "samples/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}/synthetic_ifn_run_${i}.h5ad" \
                --n_samples "${N_SAMPLES}" \
                --umap_plot "samples/fig1/task1/${cell_type}/squidiff_${GENE_SIZE}_noise_${noise_level}/umap_comparison_${i}.png" \
                --data_path "data/add_gaussian_noise_output/task1_valid_CD4T_exp_noise_std_${noise_level}.h5ad" 2>&1) || true
            
            # current output
            echo "$output"
            
            # currentoutput to amount, using 
            all_outputs+="$output\n"
        done

        # --- 3 : using AWK runstats ---
        echo -e "\n"
        echo "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" '
            # AWK Start: from eval output all 
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

            # can using count, for value/ 
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
                    printf "%-40s: N/A (no data collected)\n", name;
                }
            }

            END {
                print "==================================================================";
                printf " %s (noise: %s) final stats (%d )\n", dataset, noise, num_runs;
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
        
        echo -e "\n--- Done: $cell_type | noise: $noise_level ---\n"
    done # noise level 
done # celltype 

echo "######################################################################"
echo "###   All cell types and noise levels finished.                 ###"
echo "######################################################################"
