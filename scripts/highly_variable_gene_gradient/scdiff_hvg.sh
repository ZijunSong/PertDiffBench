#!/bin/bash

# Exit immediately if a command fails, printing an error message.
trap 'echo "ERROR: A command failed. Exiting." >&2; exit 1' ERR

# --- Configuration Area ---
# Path prefix; convention: checkpoints under checkpoints/<method>/CD4T_hvg_${gene_num}, samples under samples/highly_variable_gene_gradient/<method>_${gene_num}, logs under logs/highly_variable_gene_gradient/<method>
ROOT_DIR="${ROOT_DIR:-}"

# Gene counts to process
GENE_NUMS_LIST=(6998 6000 5000 4000 3000 2000 1000)

# (train + eval) cycles
NUM_RUNS=3

# Shared script parameters
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR="${LOGDIR:-${ROOT_DIR}logs/highly_variable_gene_gradient/scdiff}"
mkdir -p "${LOGDIR}"
NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"

# Method name for CSV first column
METHOD_NAME="scDiff"

# --- Main Script Logic ---

for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   Starting scDiff pipeline for: Number of Genes = $gene_num"
    echo "######################################################################"

    # ----- Dynamic Path and Parameter Setup -----
    dataset_name="fig1_task1_CD4T_${gene_num}"
    train_fname="CD4T_train_HVG_${gene_num}.h5ad"
    valid_fname="CD4T_valid_HVG_${gene_num}.h5ad"
    CUSTOM_DATA_PATH="${ROOT_DIR}data/highly_variable_gene_gradient"

    data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
    data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${valid_fname}"

    # Collect all run outputs
    all_outputs=""

    # ----- (Train + Eval) × NUM_RUNS -----
    echo -e "\n--- Step 1: Running Training and Evaluation (Gene Count: $gene_num, Total Runs: $NUM_RUNS) ---"
    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n--- Now running iteration $i/$NUM_RUNS ---"

        # Unique postfix per run to avoid log conflicts
        run_postfix="perturbation_${NAME}_gene${gene_num}_run${i}"

        # Run training + eval (as defined in eval_perturbation.yaml)
        output=$(python src/scDiff/main.py \
            --custom_data_path "${CUSTOM_DATA_PATH}" \
            --base configs/scdiff/eval_perturbation.yaml \
            --name "${NAME}" \
            --logdir "${LOGDIR}" \
            --postfix "${run_postfix}" \
            ${OFFLINE_SETTINGS} \
            ${data_settings} 2>&1) || true

        echo "$output"
        all_outputs+="$output\n"
    done

    # Step 2: Stats (+ CSV 2x45); samples follow same convention as ddpm_hvg.sh
    sample_dir_base="${ROOT_DIR}samples/highly_variable_gene_gradient/scdiff_${gene_num}"
    mkdir -p "${sample_dir_base}"
    csv_file="${sample_dir_base}/metrics_scdiff_${dataset_name}.csv"

    echo -e "\n"
    echo -e "$all_outputs" | awk -v dataset="${dataset_name}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_file}" '
        # Capture 11 metrics from logs
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
            if (idx==1)      { n=c_pds;                 for(i=0;i<n;i++){v=pds[i];                 s+=v} }
            else if (idx==2) { n=c_mae;                 for(i=0;i<n;i++){v=mae[i];                 s+=v} }
            else if (idx==3) { n=c_des;                 for(i=0;i<n;i++){v=des[i];                 s+=v} }
            else if (idx==4) { n=c_edist;               for(i=0;i<n;i++){v=edist[i];               s+=v} }
            else if (idx==5) { n=c_mmd;                 for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
            else if (idx==6) { n=c_r2;                  for(i=0;i<n;i++){v=r2[i];                  s+=v} }
            else if (idx==7) { n=c_pearson_all;         for(i=0;i<n;i++){v=pearson_all[i];          s+=v} }
            else if (idx==8) { n=c_pearson_delta_all;   for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
            else if (idx==9) { n=c_pearson_delta_de20;  for(i=0;i<n;i++){v=pearson_delta_de20[i];   s+=v} }
            else if (idx==10){ n=c_pearson_delta_de50;  for(i=0;i<n;i++){v=pearson_delta_de50[i];   s+=v} }
            else if (idx==11){ n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i];  s+=v} }
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

        # pretty log
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
            printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
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

            # ----- CSV 2×45: Method + 11*(mean±std) + 3*11 Run values -----
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
echo "###   All processing is complete!                                  ###"
echo "######################################################################"
