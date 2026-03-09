#!/bin/bash

# Exit immediately if a command fails
set -e

# ===== Configuration =====
# List of gene numbers
GENE_NUMS_LIST=(3000 2000 1000)
# Number of (train+eval) repetitions
NUM_RUNS=3
# Method name (first column in CSV)
METHOD_NAME="scDiffusion"

# ===== Main workflow =====
for gene_num in "${GENE_NUMS_LIST[@]}"; do
    echo "######################################################################"
    echo "###   Start processing: scDiffusion, gene count = $gene_num"
    echo "######################################################################"

    # Data paths
    train_data_path="data/highly_variable_gene_gradient/CD4T_train_HVG_${gene_num}.h5ad"
    valid_data_path="data/highly_variable_gene_gradient/CD4T_valid_HVG_${gene_num}.h5ad"
    cell_type='CD4T'   # only for logs/statistics

    # Base checkpoints and sample output directories (each run will create runN under these)
    vae_ckpt_base="checkpoints/scdiffusion/vae_checkpoint/CD4T_hvg_${gene_num}"
    diff_ckpt_base="../../checkpoints/scdiffusion/diffusion_checkpoint/cd4t_hvg_${gene_num}"
    cls_ckpt_base="../../checkpoints/scdiffusion/classifier_checkpoint/2-classifier/cd4t_hvg_${gene_num}"
    sample_base="samples/highly_variable_gene_gradient/scDiffusion_${gene_num}"

    mkdir -p "${vae_ckpt_base}" "${diff_ckpt_base}" "${cls_ckpt_base}" "${sample_base}"

    # Collect outputs of all runs for statistics and CSV
    all_outputs=""

    # ===== (train+eval) × NUM_RUNS =====
    echo -e "\n--- Start (train+eval) workflow: gene count $gene_num, total $NUM_RUNS runs ---"
    # Enter VAE directory to keep original relative paths
    cd src/scDiffusion/VAE

    for (( i=1; i<=NUM_RUNS; i++ )); do
        echo -e "\n======================"
        echo -e " Run $i/$NUM_RUNS (gene count $gene_num)"
        echo -e "======================"

        # Independent save/output dirs for each run
        vae_ckpt_dir="../../${vae_ckpt_base}/run${i}"
        diff_ckpt_dir="${diff_ckpt_base}/run${i}"
        cls_ckpt_dir="${cls_ckpt_base}/run${i}"
        sample_dir="${sample_base}/run${i}"

        mkdir -p "${vae_ckpt_dir}" "${diff_ckpt_dir}" "${cls_ckpt_dir}" "${sample_dir}"

        # Model file paths (according to your original naming scheme)
        vae_model_file="${vae_ckpt_dir}/model_seed=0_step=9999.pt"
        diff_model_file="${diff_ckpt_dir}/my_diffusion/model010000.pt"
        cls_model_file="${cls_ckpt_dir}/model009999.pt"

        # —— Step A: Train VAE —— 
        echo -e "\n--- Step A: Train VAE (genes: $gene_num, run $i) ---"
        python VAE_train.py \
            --data_dir "../../../${train_data_path}" \
            --num_genes "$gene_num" \
            --state_dict "../../../checkpoints/annotation_model_v1" \
            --save_dir "../${vae_ckpt_dir}"

        # Go back to src/scDiffusion for next steps
        cd ..

        # —— Step B: Train diffusion backbone —— 
        echo -e "\n--- Step B: Train diffusion backbone (genes: $gene_num, run $i) ---"
        python cell_train.py \
            --data_dir "../../${train_data_path}" \
            --vae_path "${vae_model_file}" \
            --save_dir "${diff_ckpt_dir}"

        # —— Step C: Train classifier —— 
        echo -e "\n--- Step C: Train classifier (genes: $gene_num, run $i) ---"
        python classifier_train.py \
            --data_dir "../../${train_data_path}" \
            --vae_path "${vae_model_file}" \
            --model_path "${cls_ckpt_dir}"

        # —— Step D: Perturbation prediction & evaluation —— 
        echo -e "\n--- Step D: Perturbation prediction & evaluation (genes: $gene_num, run $i) ---"
        output=$(python classifier_sample.py \
            --num_samples 278 \
            --train-data-path "../../${train_data_path}" \
            --model_path "${diff_model_file}" \
            --classifier_path "${cls_model_file}" \
            --ae_dir "${vae_model_file}" \
            --num_gene "$gene_num" \
            --sample_dir "../../${sample_dir}" \
            --init_cell_path "../../${valid_data_path}" 2>&1) || true

        echo "$output"
        all_outputs+="$output\n"

        # Back to VAE directory for next iteration
        cd VAE
    done

    cd ../../..

    # ===== Statistics & CSV export (2×45) =====
    # Note: CSV header row:
    #  method + 11 “mean±std” + each run’s 11 metrics (total 1+11+33 = 45 cols)
    csv_file="${sample_base}/metrics_scdiffusion_gene_${gene_num}.csv"
    echo "Absolute path of csv_file: $(realpath "$csv_file")"

    echo -e "\n"
    echo -e "$all_outputs" | awk -v dataset="$cell_type" -v runs="$NUM_RUNS" -v method="$METHOD_NAME" -v outpath="$csv_file" '
        # —— Extract 11 metrics from log —— 
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

        # —— Compute mean and std —— 
        function mean_std(idx,    i, n, s, mu, ss, v) {
            if (idx==1)      { n=c_pds;                 for(i=0;i<n;i++){v=pds[i];                 s+=v} }
            else if (idx==2) { n=c_mae;                 for(i=0;i<n;i++){v=mae[i];                 s+=v} }
            else if (idx==3) { n=c_des;                 for(i=0;i<n;i++){v=des[i];                 s+=v} }
            else if (idx==4) { n=c_edist;               for(i=0;i<n;i++){v=edist[i];               s+=v} }
            else if (idx==5) { n=c_mmd;                 for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
            else if (idx==6) { n=c_r2;                  for(i=0;i<n;i++){v=r2[i];                  s+=v} }
            else if (idx==7) { n=c_pearson_all;         for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
            else if (idx==8) { n=c_pearson_delta_all;   for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
            else if (idx==9) { n=c_pearson_delta_de20;  for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
            else if (idx==10){ n=c_pearson_delta_de50;  for(i=0;i<n;i++){v=pearson_delta_de50[i];  s+=v} }
            else if (idx==11){ n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
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

        # —— Get j-th value of metric —— 
        function get_val(idx, j,    v){
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

        # —— Print stats to console —— 
        function print_stats(name, arr, n,    i, s, mu, ss, std) {
            if (n > 0) {
                for (i = 0; i < n; i++) s += arr[i];
                mu = s / n;
                for (i = 0; i < n; i++) ss += (arr[i] - mu)^2;
                std = (n > 1) ? sqrt(ss / (n - 1)) : 0;
                printf "%-40s: %.4f ± %.4f\n", name, mu, std;
            } else {
                printf "%-40s: N/A (no data)\n", name;
            }
        }

        END {
            print "==================================================================";
            printf " Final statistics: dataset %s (total %d runs)\n", dataset, runs;
            print "==================================================================";
            print_stats("PDS",                 pds, c_pds);
            print_stats("MAE",                 mae, c_mae);
            print_stats("DES",                 des, c_des);
            print "----------------------------------------";
            print_stats("E-Distance",          edist, c_edist);
            print_stats("MMD",                 mmd,  c_mmd);
            print_stats("R2",                  r2,   c_r2);
            print "----------------------------------------";
            print_stats("Pearson (all genes)", pearson_all,         c_pearson_all);
            print_stats("Pearson-Delta (all genes)", pearson_delta_all,   c_pearson_delta_all);
            print_stats("Pearson-Delta (top 20 DE genes)", pearson_delta_de20,  c_pearson_delta_de20);
            print_stats("Pearson-Delta (top 50 DE genes)", pearson_delta_de50,  c_pearson_delta_de50);
            print_stats("Pearson-Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100);
            print "==================================================================\n";

            # ===== Generate 2×45 CSV =====
            metric_name[1]="PDS";
            metric_name[2]="MAE";
            metric_name[3]="DES";
            metric_name[4]="E-Distance";
            metric_name[5]="MMD";
            metric_name[6]="R2";
            metric_name[7]="Pearson (all genes)";
            metric_name[8]="Pearson-Delta (all genes)";
            metric_name[9]="Pearson-Delta (top20 DE genes)";
            metric_name[10]="Pearson-Delta (top50 DE genes)";
            metric_name[11]="Pearson-Delta (top100 DE genes)";

            # Header row
            header = "Method";
            for (i=1;i<=11;i++) header = header "," metric_name[i] "(mean±std)";
            for (r=1;r<=runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_name[i];

            # Data row
            row = method;
            for (i=1;i<=11;i++) {
                ms = mean_std(i); split(ms, p, "|");
                row = row sprintf(",%.4f±%.4f", p[1], p[2]);
            }
            for (r=0;r<runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", get_val(i, r));

            print header > outpath;
            print row   >> outpath;
            close(outpath);
            printf("CSV written to: %s\n", outpath);
        }
    '

    echo -e "\n--- Done: cell type = $cell_type ---\n"
done

echo "######################################################################"
echo "###   All cell types processed successfully                         ###"
echo "######################################################################"
