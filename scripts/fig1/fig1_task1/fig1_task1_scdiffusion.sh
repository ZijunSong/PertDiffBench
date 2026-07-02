#!/bin/bash

# Exit on error; print a clear message
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# -------------------- Configuration --------------------
# Path prefix; convention: data under data/highly_variable_gene_gradient/; checkpoints under CKPT_ROOT/fig1/task1/scdiffusion/.../<cell_type>_1000; samples under samples/fig1/task1/<cell_type>/<method>_1000; logs under logs/fig1_task1
ROOT_DIR="${ROOT_DIR:-/data/ppnm/data/PertDiffBench/}"
CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"
# scDiffusion external annotation model (e.g. /data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1)
ANNOTATION_MODEL_DIR="${ANNOTATION_MODEL_DIR:-/data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR="${LOGDIR:-${ROOT_DIR}logs/fig1_task1}"
NUM_GENES="${NUM_GENES:-1000}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiffusion}"   # Method name (first column in CSV)
# -------------------------------------------------------

# Define cell types
CELL_TYPES=(
  'B'
  'CD4T'
  'CD8T'
  'CD14+Mono'
  'Dendritic'
  'FCGR3A+Mono'
  'NK'
)

# n_samples per cell type (max paired cells in valid set)
source "scripts/lib/max_n_samples.sh"
declare -A SAMPLES_MAP=()
build_samples_map_from_valid_h5ad "${ROOT_DIR}" "${CELL_TYPES[@]}"

mkdir -p "${LOGDIR}"

for cell_type in "${CELL_TYPES[@]}"; do
  echo "######################################################################"
  echo "###   Starting full pipeline for cell type: ${cell_type} (${NUM_RUNS} runs)"
  echo "######################################################################"

  num_samples=${SAMPLES_MAP[$cell_type]}
  [ -z "$num_samples" ] && { echo "No n_samples configured for ${cell_type}"; exit 1; }
  echo "### Using num_samples: ${num_samples}"

  # Paths (same convention across fig1 task1 scripts; scdiffusion uses subdirs under checkpoints/scdiffusion/)
  train_h5="${ROOT_DIR}data/highly_variable_gene_gradient/${cell_type}_train_HVG_${NUM_GENES}.h5ad"
  valid_h5="${ROOT_DIR}data/highly_variable_gene_gradient/${cell_type}_valid_HVG_${NUM_GENES}.h5ad"
  vae_base="${CKPT_ROOT}/fig1/task1/scdiffusion/vae_checkpoint/${cell_type}_${NUM_GENES}"
  diff_base="${CKPT_ROOT}/fig1/task1/scdiffusion/diffusion_checkpoint/${cell_type}_${NUM_GENES}"
  cls_base="${CKPT_ROOT}/fig1/task1/scdiffusion/classifier_checkpoint/${cell_type}_${NUM_GENES}"
  sample_dir_base="${ROOT_DIR}samples/fig1/task1/${cell_type}/scDiffusion_1000"
  mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_dir_base}"
  csv_path="${sample_dir_base}/metrics_${METHOD_NAME}_${cell_type}_hvg_${NUM_GENES}.csv"
  log_file="${LOGDIR}/scdiffusion_${cell_type}_hvg_${NUM_GENES}.log"

  {
    echo "== $(date '+%F %T') | cell_type=${cell_type} genes=${NUM_GENES} runs=${NUM_RUNS} n_samples=${num_samples} =="

    all_outputs=""

    # 3x runs: train (VAE + Diffusion + Classifier) + sample/eval
    for (( i=1; i<=NUM_RUNS; i++ )); do
      export RUN_SEED=$(($i-1))
      echo
      echo "======================"
      echo " Run ${i}/${NUM_RUNS} for ${cell_type}"
      echo "======================"

      # Per-run directories
      vae_dir="${vae_base}/run${i}"
      diff_dir="${diff_base}/run${i}"
      cls_dir="${cls_base}/run${i}"
      sample_dir_run="${sample_dir_base}/run${i}"
      mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${sample_dir_run}"

      vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
      diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
      cls_ckpt="${cls_dir}/model009999.pt"

      [ -n "$ROOT_DIR" ] && train_h5_vae="${train_h5}" || train_h5_vae="../../../${train_h5}"
      [ -n "$ROOT_DIR" ] && train_h5_sd="${train_h5}" || train_h5_sd="../../${train_h5}"
      [ -n "$ROOT_DIR" ] && valid_h5_arg="${valid_h5}" || valid_h5_arg="../../${valid_h5}"
      [ -n "$ROOT_DIR" ] && vae_dir_arg="${vae_dir}" || vae_dir_arg="../../../${vae_dir}"
      [ -n "$ROOT_DIR" ] && diff_dir_arg="${diff_dir}" || diff_dir_arg="../../${diff_dir}"
      [ -n "$ROOT_DIR" ] && cls_dir_arg="${cls_dir}" || cls_dir_arg="../../${cls_dir}"
      [ -n "$ROOT_DIR" ] && vae_ckpt_arg="${vae_ckpt}" || vae_ckpt_arg="../../${vae_ckpt}"
      [ -n "$ROOT_DIR" ] && diff_ckpt_arg="${diff_ckpt}" || diff_ckpt_arg="../../${diff_ckpt}"
      [ -n "$ROOT_DIR" ] && cls_ckpt_arg="${cls_ckpt}" || cls_ckpt_arg="../../${cls_ckpt}"
      [ -n "$ROOT_DIR" ] && sample_run_arg="${sample_dir_run}" || sample_run_arg="../../${sample_dir_run}"

      # Step 1: Train the Autoencoder (VAE)
      echo
      echo "--- Step 1: Training VAE for ${cell_type} [run ${i}] ---"
      pushd src/scDiffusion/VAE >/dev/null
      python VAE_train.py \
        --data_dir "${train_h5_vae}" \
        --num_genes "${NUM_GENES}" \
        --state_dict "${ANNOTATION_MODEL_DIR}" \
        --save_dir "${vae_dir_arg}"
      popd >/dev/null

      # Step 2: Train the diffusion backbone
      echo
      echo "--- Step 2: Training Diffusion for ${cell_type} [run ${i}] ---"
      pushd src/scDiffusion >/dev/null
      python cell_train.py \
        --data_dir "${train_h5_sd}" \
        --vae_path "${vae_ckpt_arg}" \
        --save_dir "${diff_dir_arg}"
      popd >/dev/null

      # Step 3: Train the classifier
      echo
      echo "--- Step 3: Training Classifier for ${cell_type} [run ${i}] ---"
      pushd src/scDiffusion >/dev/null
      python classifier_train.py \
        --data_dir "${train_h5_sd}" \
        --vae_path "${vae_ckpt_arg}" \
        --model_path "${cls_dir_arg}"
      popd >/dev/null

      # Step 4: Sampling and Evaluation
      echo
      echo "--- Step 4: Sampling & Evaluation for ${cell_type} [run ${i}] ---"
      pushd src/scDiffusion >/dev/null
      output=$(
        python classifier_sample.py \
          --num_samples "${num_samples}" \
          --train-data-path "${train_h5_sd}" \
          --model_path "${diff_ckpt_arg}" \
          --classifier_path "${cls_ckpt_arg}" \
          --ae_dir "${vae_ckpt_arg}" \
          --num_gene "${NUM_GENES}" \
          --sample_dir "${sample_run_arg}" \
          --out_h5ad "${sample_run_arg}/synthetic_ifn_${i}.h5ad" \
          --umap_plot "${sample_run_arg}/umap_comparison_${i}.png" \
          --init_cell_path "${valid_h5_arg}" 2>&1
      ) || true
      popd >/dev/null

      echo "$output"
      all_outputs+="$output\n"
    done

    # ========== Aggregate stats + write CSV ==========
    echo
    echo -e "$all_outputs" | awk -v dataset="${cell_type}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
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

      # mean|std helper
      function mean_std(idx,    i,n,s,mu,ss,v) {
        if (idx==1)  { n=c_pds;                  for(i=0;i<n;i++){v=pds[i];                 s+=v} }
        else if(idx==2){ n=c_mae;                for(i=0;i<n;i++){v=mae[i];                 s+=v} }
        else if(idx==3){ n=c_des;                for(i=0;i<n;i++){v=des[i];                 s+=v} }
        else if(idx==4){ n=c_edist;              for(i=0;i<n;i++){v=edist[i];               s+=v} }
        else if(idx==5){ n=c_mmd;                for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
        else if(idx==6){ n=c_r2;                 for(i=0;i<n;i++){v=r2[i];                  s+=v} }
        else if(idx==7){ n=c_pearson_all;        for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
        else if(idx==8){ n=c_pearson_delta_all;  for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
        else if(idx==9){ n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
        else if(idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i];  s+=v} }
        else if(idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
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

      # j-th run raw value (0-based)
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

      function print_stat(name, data, count,    i,s,mu,ss,std) {
        if (count > 0) {
          for (i = 0; i < count; i++) s += data[i];
          mu = s / count;
          for (i = 0; i < count; i++) ss += (data[i] - mu)^2;
          std = (count > 1) ? sqrt(ss / (count - 1)) : 0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std;
        } else {
          printf "%-40s: N/A (No data)\n", name;
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

        # CSV header: Method + 11(mean±std) + raw(3x11)
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
        for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
        for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));

        print header > csv_path;
        print row    >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    '

    echo
    echo "--- Finished pipeline for cell type: ${cell_type} ---"
    echo
  } 2>&1 | tee -a "${log_file}"

done

echo "######################################################################"
echo "###   All cell type processing is complete!                        ###"
echo "######################################################################"
