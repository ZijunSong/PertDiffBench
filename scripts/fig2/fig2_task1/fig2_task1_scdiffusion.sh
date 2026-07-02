#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
IFS=$'\n\t'
trap 'echo ERROR && exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

# -------------------- Config --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GENES="${NUM_GENES:-3000}" # can : HVG count
NUM_RUNS=3                         # fixed 3 train+eval runs

# W&B ( )
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# ---------------- Project Root ------------------
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "$HOMEDIR"
echo "PWD: $(pwd)"

# ---------------- Datasets ----------------------
SEEDS=('123' '345' '567')

# CSV ( all seed)
GLOBAL_DIR="samples/fig2/task1/scDiffusion"
mkdir -p "${GLOBAL_DIR}"
GLOBAL_CSV="${GLOBAL_DIR}/metrics_all.csv"

# header
if [[ ! -f "${GLOBAL_CSV}" ]]; then
  {
    printf "Dataset,Method"
    printf ",PDS (mean±std),MAE (mean±std),DES (mean±std),E-Distance (mean±std),MMD (mean±std),R2 (mean±std)"
    printf ",Pearson (all genes) (mean±std),Pearson Delta (all genes) (mean±std)"
    printf ",Pearson Delta (top 20 DE genes) (mean±std),Pearson Delta (top 50 DE genes) (mean±std),Pearson Delta (top 100 DE genes) (mean±std)"
    for r in 1 2 3; do
      printf ",Run%d PDS,Run%d MAE,Run%d DES,Run%d E-Distance,Run%d MMD,Run%d R2" $r $r $r $r $r $r
      printf ",Run%d Pearson (all genes),Run%d Pearson Delta (all genes)" $r $r
      printf ",Run%d Pearson Delta (top 20 DE genes),Run%d Pearson Delta (top 50 DE genes),Run%d Pearson Delta (top 100 DE genes)" $r $r $r
    done
    printf "\n"
  } > "${GLOBAL_CSV}"
fi

# ---------------- Main Loop ---------------------
for seed in "${SEEDS[@]}"; do
  dataset_base="seed${seed}"
  train_h5="data/fig2/task1_unseen_pert/${dataset_base}_control_train.h5ad"
  valid_h5="data/fig2/task1_unseen_pert/${dataset_base}_control_test.h5ad"
  N_SAMPLES="$(max_n_samples_paired "${valid_h5}")"

  echo "######################################################################"
  echo "###   Full pipeline for ${dataset_base} (${NUM_RUNS} runs, ${NUM_GENES} HVG)"
  echo "######################################################################"

  # each seed output dir ( original )
  vae_base="checkpoints/scdiffusion/vae_checkpoint/fig2_task1/${dataset_base}_${NUM_GENES}"
  diff_base="checkpoints/scdiffusion/diffusion_checkpoint/fig2_task1/${dataset_base}_${NUM_GENES}"
  cls_base="checkpoints/scdiffusion/classifier_checkpoint/2-classifier/fig2_task1/${dataset_base}_${NUM_GENES}"
  sample_base="samples/fig2/task1_unseen_pert/${dataset_base}/scDiffusion_${NUM_GENES}"
  mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_base}"

  ALL_OUTPUTS=""

  # ============ (train+eval) ============
  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for ${dataset_base}"
    echo "======================"

    # each run run standalonesubdir, 
    vae_dir="${vae_base}/run${i}"
    diff_dir="${diff_base}/run${i}"
    cls_dir="${cls_base}/run${i}"
    run_sample_dir="${sample_base}/run${i}"
    mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${run_sample_dir}"

    # expects checkpoint filename (andoriginal stay consistent)
    vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
    diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
    cls_ckpt="${cls_dir}/model009999.pt"

    # ---- Step 1: VAE train ----
    echo "--- Step 1: Training VAE ..."
    pushd src/scDiffusion/VAE >/dev/null
    python VAE_train.py \
      --data_dir "../../../${train_h5}" \
      --num_genes "${NUM_GENES}" \
      --state_dict ../../../checkpoints/annotation_model_v1 \
      --save_dir "../../../${vae_dir}"
    popd >/dev/null

    # ---- Step 2: Diffusion train ----
    echo "--- Step 2: Training Diffusion ..."
    pushd src/scDiffusion >/dev/null
    python cell_train.py \
      --data_dir "../../${train_h5}" \
      --vae_path "../../${vae_ckpt}" \
      --save_dir "../../${diff_dir}"
    popd >/dev/null

    # ---- Step 3: Classifier train ----
    echo "--- Step 3: Training Classifier ..."
    pushd src/scDiffusion >/dev/null
    python classifier_train.py \
      --data_dir "../../${train_h5}" \
      --vae_path "../../${vae_ckpt}" \
      --model_path "../../${cls_dir}"
    popd >/dev/null

    # ---- Step 4: andeval ----
    echo "--- Step 4: Sampling & Evaluation ..."
    pushd src/scDiffusion >/dev/null
    run_out="$(
      python classifier_sample.py \
        --num_samples "${N_SAMPLES}" \
        --train-data-path "../../${train_h5}" \
        --model_path "../../${diff_ckpt}" \
        --classifier_path "../../${cls_ckpt}" \
        --ae_dir "../../${vae_ckpt}" \
        --num_gene "${NUM_GENES}" \
        --sample_dir "../../${run_sample_dir}" \
        --out_h5ad "../../${run_sample_dir}/synthetic_ifn_${i}.h5ad" \
        --umap_plot "../../${run_sample_dir}/umap_comparison_${i}.png" \
        --init_cell_path "../../${valid_h5}" 2>&1
    )" || true
    popd >/dev/null

    echo "${run_out}"
    # only canparse 
    ALL_OUTPUTS+="$(
      printf "%s\n" "${run_out}" | grep -E \
        "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
        || true
    )"
    ALL_OUTPUTS+=$'\n'
  done

  # ============ to CSV ============
  printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="${dataset_base}_control_test" -v num_runs="${NUM_RUNS}" -v method="scDiffusion(${NUM_GENES})" -v csv_path="${GLOBAL_CSV}" '
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
    /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
    /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
    /^E-Distance:/                              { edist[c_edist++] = to_num($NF) }
    /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++]  = to_num($NF) }
    /R-squared \(R2\):/                         { r2[c_r2++]    = to_num($NF) }
    /Pearson \(all genes\):/                    { p_all[c_p_all++] = to_num($NF) }
    /Pearson Delta \(all genes\):/              { pd_all[c_pd_all++] = to_num($NF) }
    /Pearson Delta \(top 20 DE genes\):/        { pd20[c_pd20++] = to_num($NF) }
    /Pearson Delta \(top 50 DE genes\):/        { pd50[c_pd50++] = to_num($NF) }
    /Pearson Delta \(top 100 DE genes\):/       { pd100[c_pd100++] = to_num($NF) }

    function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n : 0 }
    function std(a,n,  mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }

    function mean_std(idx,  n,mu,sd){
      if(idx==1){ n=c_pds;    mu=mean(pds,n);    sd=std(pds,n) }
      else if(idx==2){ n=c_mae;    mu=mean(mae,n);    sd=std(mae,n) }
      else if(idx==3){ n=c_des;    mu=mean(des,n);    sd=std(des,n) }
      else if(idx==4){ n=c_edist;  mu=mean(edist,n);  sd=std(edist,n) }
      else if(idx==5){ n=c_mmd;    mu=mean(mmd,n);    sd=std(mmd,n) }
      else if(idx==6){ n=c_r2;     mu=mean(r2,n);     sd=std(r2,n) }
      else if(idx==7){ n=c_p_all;  mu=mean(p_all,n);  sd=std(p_all,n) }
      else if(idx==8){ n=c_pd_all; mu=mean(pd_all,n); sd=std(pd_all,n) }
      else if(idx==9){ n=c_pd20;   mu=mean(pd20,n);   sd=std(pd20,n) }
      else if(idx==10){n=c_pd50;   mu=mean(pd50,n);   sd=std(pd50,n) }
      else if(idx==11){n=c_pd100;  mu=mean(pd100,n);  sd=std(pd100,n) }
      return sprintf("%.6f±%.6f", mu, sd)
    }

    function val(idx, r, v){
      if(idx==1) v=pds[r];
      else if(idx==2) v=mae[r];
      else if(idx==3) v=des[r];
      else if(idx==4) v=edist[r];
      else if(idx==5) v=mmd[r];
      else if(idx==6) v=r2[r];
      else if(idx==7) v=p_all[r];
      else if(idx==8) v=pd_all[r];
      else if(idx==9) v=pd20[r];
      else if(idx==10) v=pd50[r];
      else if(idx==11) v=pd100[r];
      return (v=="") ? 0 : v;
    }

    END {
      row=ds "," method;
      for(i=1;i<=11;i++) row=row "," mean_std(i);
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

      print row >> csv_path;
      close(csv_path);
      printf("CSV appended: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished ${dataset_base} ---\n"
done

echo "######################################################################"
echo "###   All seeds completed!  CSV @ ${GLOBAL_CSV}"
echo "######################################################################"
