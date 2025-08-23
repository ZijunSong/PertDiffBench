#!/bin/bash

# Set to exit immediately if a command exits with a non-zero status.
set -e

# Define an array of all cell types to be processed
CELL_TYPES=(
    # 'B'
    'CD4T'
    # 'CD8T'
    # 'CD14+Mono'
    # 'Dendritic'
    # 'FCGR3A+Mono'
    # 'NK'
)

# Define common parameters
NUM_GENES="6000"
NUM_RUNS=3

# Loop through each cell type
for cell_type in "${CELL_TYPES[@]}"; do
    echo "######################################################################"
    echo "###   Starting full pipeline for cell type: $cell_type"
    echo "######################################################################"

    # --- Step 1: Train the Autoencoder ---
    echo -e "\n--- Step 1: Training VAE for $cell_type ---"
    cd src/scDiffusion/VAE
    python VAE_train.py \
        --data_dir "../../../data/fig1/task1_hvg_output/${cell_type}_train_HVG_${NUM_GENES}.h5ad" \
        --num_genes "$NUM_GENES" \
        --state_dict ../../../checkpoints/scimilarity/model_v1.1 \
        --save_dir "../../../checkpoints/scdiffusion/vae_checkpoint/task1/${cell_type}_${NUM_GENES}"
    cd ../../.. # Return to project root

    # --- Step 2: Train the diffusion backbone ---
    echo -e "\n--- Step 2: Training Diffusion model for $cell_type ---"
    cd src/scDiffusion
    python cell_train.py \
        --data_dir "../../data/fig1/task1_hvg_output/${cell_type}_train_HVG_${NUM_GENES}.h5ad" \
        --vae_path "../../checkpoints/scdiffusion/vae_checkpoint/task1/${cell_type}_${NUM_GENES}/model_seed=0_step=9999.pt" \
        --save_dir "../../checkpoints/scdiffusion/diffusion_checkpoint/task1/${cell_type}_${NUM_GENES}"
    cd ../.. # Return to project root

    # --- Step 3: Train the classifier ---
    echo -e "\n--- Step 3: Training Classifier for $cell_type ---"
    cd src/scDiffusion
    python classifier_train.py \
        --data_dir "../../data/fig1/task1_hvg_output/${cell_type}_train_HVG_${NUM_GENES}.h5ad" \
        --vae_path "../../checkpoints/scdiffusion/vae_checkpoint/task1/${cell_type}_${NUM_GENES}/model_seed=0_step=9999.pt" \
        --model_path "../../checkpoints/scdiffusion/classifier_checkpoint/2-classifier/task1/${cell_type}_${NUM_GENES}"
    cd ../.. # Return to project root

    # --- Step 4: Predict the perturbation (run multiple times) ---
    echo -e "\n--- Step 4: Sampling & Evaluation for $cell_type ($NUM_RUNS runs) ---"
    cd src/scDiffusion
    
    all_outputs=""
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Running sampling iteration $i/$NUM_RUNS for $cell_type ---"
        output=$(python classifier_sample.py \
            --num_samples 100 \
            --train-data-path "../../data/fig1/task1_hvg_output/${cell_type}_train_HVG_${NUM_GENES}.h5ad" \
            --model_path "../../checkpoints/scdiffusion/diffusion_checkpoint/task1/${cell_type}_${NUM_GENES}/my_diffusion/model010000.pt" \
            --classifier_path "../../checkpoints/scdiffusion/classifier_checkpoint/2-classifier/task1/${cell_type}_${NUM_GENES}/model009999.pt" \
            --ae_dir "../../checkpoints/scdiffusion/vae_checkpoint/task1/${cell_type}_${NUM_GENES}/model_seed=0_step=9999.pt" \
            --num_gene "$NUM_GENES" \
            --sample_dir "../../samples/fig1/task1/${cell_type}_${NUM_GENES}/scDiffusion" \
            --out_h5ad "../../samples/fig1/task1/${cell_type}_${NUM_GENES}/scDiffusion/synthetic_ifn_${i}.h5ad" \
            --umap_plot "../../samples/fig1/task1/${cell_type}_${NUM_GENES}/scDiffusion/umap_comparison_${i}.png" \
            --init_cell_path "../../data/fig1/task1_hvg_output/${cell_type}_valid_HVG_${NUM_GENES}.h5ad" 2>&1) || true
        
        echo "$output"
        all_outputs+="$output\n"
    done
    cd ../.. # Return to project root

    # --- Step 5: Statistical Calculation using AWK ---
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
