#!/bin/bash

# Exit immediately if a command fails
set -e

# --- Configuration Area ---

# Gene counts to process
GENE_NUMS_LIST=(6998 6000 5000 4000 3000 2000 1000)

# How many (train + eval) cycles
NUM_RUNS=3

# Config file
CONFIG_FILE="configs/baselines/mlp_ddpm_mlp.yaml"

# Method name for CSV first column
METHOD_NAME="DDPM+MLP"

# --- Main Script Logic ---

for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   Starting pipeline for: Gene Count = $gene_num"
    echo "######################################################################"

    # Data paths
    train_data_path="data/highly_variable_gene_gradient/CD4T_train_HVG_${gene_num}.h5ad"
    valid_data_path="data/highly_variable_gene_gradient/CD4T_valid_HVG_${gene_num}.h5ad"

    # Base output dirs
    save_dir_base="checkpoints/ddpm_mlp/CD4T_hvg_${gene_num}"
    sample_dir_base="samples/highly_variable_gene_gradient/mlp_ddpm_mlp_${gene_num}"
    mkdir -p "$save_dir_base" "$sample_dir_base"

    # Aggregate all eval outputs across runs
    all_outputs=""

    # --- (Train + Eval) cycles ---
    for (( run_idx=1; run_idx<=NUM_RUNS; run_idx++ )); do
        echo -e "\n======================"
        echo -e " Run ${run_idx}/${NUM_RUNS} for Gene Count ${gene_num}"
        echo -e "======================"

        save_dir_run="${save_dir_base}/run${run_idx}"
        sample_dir_run="${sample_dir_base}/run${run_idx}"
        mkdir -p "${save_dir_run}" "${sample_dir_run}"

        checkpoint_file="${save_dir_run}/model_epoch_1000.pth"

        # Step A: Train
        echo -e "\n--- Step A: Training (Gene Count: $gene_num, Run: $run_idx) ---"
        python scripts/baseline/train_mlp_ddpm_mlp.py \
            --config "$CONFIG_FILE" \
            --data-path "$train_data_path" \
            --save-weight-dir "$save_dir_run" \
            --gene-nums "$gene_num"

        # Step B: Eval (right after training)
        echo -e "\n--- Step B: Evaluating (Gene Count: $gene_num, Run: $run_idx) ---"
        output=$(python scripts/baseline/eval_mlp_ddpm_mlp.py \
            --config "$CONFIG_FILE" \
            --train-data-path "$train_data_path" \
            --data-path "$valid_data_path" \
            --ckpt "$checkpoint_file" \
            --out_h5ad "${sample_dir_run}/synthetic_ifn_run${run_idx}.h5ad" \
            --n_samples 278 \
            --gene-nums "$gene_num" 2>&1) || true

        echo "$output"
        all_outputs+="$output\n"
    done

    # --- Stats + CSV (2x45) ---
    csv_out_dir="${sample_dir_base}"
    mkdir -p "${csv_out_dir}"
    csv_file="${csv_out_dir}/metrics_mlp_ddpm_mlp_gene_${gene_num}.csv"

    echo -e "\n"
    echo -e "$all_outputs" | awk -v gene_count="$gene_num" -v num_runs="$NUM_RUNS" -v method="$METHOD_NAME" -v csv_path="$csv_file" '
        # Capture 11 metrics
        /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
        /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = $NF }
        /Differential Expression Score \(DES\):/    { des[c_des++] = $NF }
        /E-Distance:/                               { edist[c_edist++] = $NF }
        /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = $NF }
        /R-squared \(R2\):/                         { r2[c_r2++] = $NF }
        /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++] = $NF }
        /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = $NF }
        /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
        /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
        /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

        # mean|std for metric idx
        function mean_std(idx,    i, n, s, mu, ss, v) {
            if (idx==1)      { n=c_pds;              for(i=0;i<n;i++){v=pds[i];                    s+=v} }
            else if (idx==2) { n=c_mae;              for(i=0;i<n;i++){v=mae[i];                    s+=v} }
            else if (idx==3) { n=c_des;              for(i=0;i<n;i++){v=des[i];                    s+=v} }
            else if (idx==4) { n=c_edist;            for(i=0;i<n;i++){v=edist[i];                  s+=v} }
            else if (idx==5) { n=c_mmd;              for(i=0;i<n;i++){v=mmd[i];                    s+=v} }
            else if (idx==6) { n=c_r2;               for(i=0;i<n;i++){v=r2[i];                     s+=v} }
            else if (idx==7) { n=c_pearson_all;      for(i=0;i<n;i++){v=pearson_all[i];            s+=v} }
            else if (idx==8) { n=c_pearson_delta_all;for(i=0;i<n;i++){v=pearson_delta_all[i];      s+=v} }
            else if (idx==9) { n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];    s+=v} }
            else if (idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i];    s+=v} }
            else if (idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i];  s+=v} }
            mu = (n>0)? s/n : 0;
            for(i=0;i<n;i++){
                if (idx==1) v=pds[i];
                else if(idx==2) v=mae[i];
                else if(idx==3) v=des[i];
                else if(idx==4) v=edist[i];
                else if(idx==5) v=mmd[i];
                else if(idx==6) v=r2[i];
                else if(idx==7) v=pearson_all[i];
                else if(idx==8) v=pearson_delta_all[i];
                else if(idx==9) v=pearson_delta_de20[i];
                else if(idx==10) v=pearson_delta_de50[i];
                else if(idx==11) v=pearson_delta_de100[i];
                ss += (v - mu) * (v - mu);
            }
            return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
        }

        # value of metric idx at run j (0-based)
        function val(idx, j,    v){
            if (idx==1) v=pds[j];
            else if(idx==2) v=mae[j];
            else if(idx==3) v=des[j];
            else if(idx==4) v=edist[j];
            else if(idx==5) v=mmd[j];
            else if(idx==6) v=r2[j];
            else if(idx==7) v=pearson_all[j];
            else if(idx==8) v=pearson_delta_all[j];
            else if(idx==9) v=pearson_delta_de20[j];
            else if(idx==10) v=pearson_delta_de50[j];
            else if(idx==11) v=pearson_delta_de100[j];
            return v;
        }

        # pretty log (optional)
        function print_stat(name, data, count,    i, s, mu, ss, std) {
            if (count > 0) {
                for (i = 0; i < count; i++) s += data[i];
                mu = s / count;
                for (i = 0; i < count; i++) ss += (data[i] - mu)^2;
                std = (count > 1) ? sqrt(ss / (count - 1)) : 0;
                printf "%-40s: %.4f ± %.4f\n", name, mu, std;
            } else {
                printf "%-40s: N/A (No data collected)\n", name;
            }
        }

        END {
            print "==================================================================";
            printf " Final statistics for Gene Count %s (%d runs: train+eval)\n", gene_count, num_runs;
            print "==================================================================";
            print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
            print_stat("Mean Absolute Error (MAE)",       mae, c_mae);
            print_stat("Differential Expression Score (DES)", des, c_des);
            print "----------------------------------------";
            print_stat("E-Distance",                        edist, c_edist);
            print_stat("Maximum Mean Discrepancy (MMD)",    mmd,  c_mmd);
            print_stat("R-squared (R2)",                    r2,   c_r2);
            print "----------------------------------------";
            print_stat("Pearson (all genes)",               pearson_all,         c_pearson_all);
            print_stat("Pearson Delta (all genes)",         pearson_delta_all,   c_pearson_delta_all);
            print_stat("Pearson Delta (top 20 DE genes)",   pearson_delta_de20,  c_pearson_delta_de20);
            print_stat("Pearson Delta (top 50 DE genes)",   pearson_delta_de50,  c_pearson_delta_de50);
            print_stat("Pearson Delta (top 100 DE genes)",  pearson_delta_de100, c_pearson_delta_de100);
            print "==================================================================\n";

            # ---- CSV (2x45): Method + 11*(mean±std) + 3*11 run values ----
            metric_names[1]="PDS";
            metric_names[2]="MAE";
            metric_names[3]="DES";
            metric_names[4]="E-Distance";
            metric_names[5]="MMD";
            metric_names[6]="R2";
            metric_names[7]="Pearson (all genes)";
            metric_names[8]="Pearson Delta (all genes)";
            metric_names[9]="Pearson Delta (top 20 DE genes)";
            metric_names[10]="Pearson Delta (top 50 DE genes)";
            metric_names[11]="Pearson Delta (top 100 DE genes)";

            header = "Method";
            for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
            for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];

            row = method;
            for (i=1;i<=11;i++) {
                ms = mean_std(i); split(ms, p, "|");
                row = row sprintf(",%.4f±%.4f", p[1], p[2]);
            }
            for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));

            print header > csv_path;
            print row    >> csv_path;
            close(csv_path);
            printf("CSV written: %s\n", csv_path);
        }
    '

    echo -e "\n--- Finished pipeline for gene count: $gene_num ---\n"
done

echo "######################################################################"
echo "###   All gene count processing is complete!                       ###"
echo "######################################################################"
