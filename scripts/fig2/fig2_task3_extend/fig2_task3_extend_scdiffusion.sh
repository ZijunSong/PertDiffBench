#!/usr/bin/env bash
# Leave-one-out: hold one species for test, merge the rest for training.
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
IFS=$'\n\t'
trap 'echo "[ERROR] command failed — abort." >&2' ERR
export LC_ALL=C LC_NUMERIC=C

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

NUM_GENES="${NUM_GENES:-6619}"
NUM_RUNS="${NUM_RUNS:-3}"
ALL_SPECIES=( "mouse" "pig" "rabbit" "rat" )

# Unified data and checkpoint roots (override via env if needed)
DATA_ROOT="${DATA_ROOT:-/data/ppnm/data/PertDiffBench/data/fig2_task3_cross_species}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
HOMEDIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$HOMEDIR"
echo "PWD: $(pwd)"

mkdir -p "$DATA_ROOT"

# Global CSV for all leave-one-out folds
GLOBAL_DIR="samples/fig2/task3_extend/scDiffusion_${NUM_GENES}"
mkdir -p "${GLOBAL_DIR}"
GLOBAL_CSV="${GLOBAL_DIR}/metrics_all.csv"

if [[ ! -f "${GLOBAL_CSV}" ]]; then
  {
    printf "Dataset,Species,Method"
    printf ",PDS (mean±std),MAE (mean±std),DES (mean±std),E-Distance (mean±std),MMD (mean±std),R2 (mean±std)"
    printf ",Pearson (all) (mean±std),PearsonΔ(all) (mean±std)"
    printf ",PearsonΔ(top20) (mean±std),PearsonΔ(top50) (mean±std),PearsonΔ(top100) (mean±std)"
    for r in 1 2 3; do
      printf ",Run%d PDS,Run%d MAE,Run%d DES,Run%d E-Distance,Run%d MMD,Run%d R2" $r $r $r $r $r $r
      printf ",Run%d Pearson(all),Run%d PearsonΔ(all)" $r $r
      printf ",Run%d PearsonΔ(top20),Run%d PearsonΔ(top50),Run%d PearsonΔ(top100)" $r $r $r
    done
    printf "\n"
  } > "${GLOBAL_CSV}"
fi

for test_species in "${ALL_SPECIES[@]}"; do
  train_species_list=()
  for s in "${ALL_SPECIES[@]}"; do
    [[ "$s" != "$test_species" ]] && train_species_list+=( "$s" )
  done
  train_species_comma=$(IFS=,; echo "${train_species_list[*]}")

  MERGED_TRAIN="${DATA_ROOT}/merged_train_${test_species}.h5ad"
  if [[ ! -f "${MERGED_TRAIN}" ]]; then
    echo "### Building merged train (leave-out ${test_species}): ${train_species_comma} -> ${MERGED_TRAIN}"
    python scripts/fig2/fig2_task3_extend/merge_species_control_ifn.py \
      --data-root "${DATA_ROOT}" \
      --train-species "${train_species_comma}" \
      --out "${MERGED_TRAIN}"
  fi

  TRAIN_H5="${MERGED_TRAIN}"
  VALID_H5="${DATA_ROOT}/${test_species}_control_ifn.h5ad"

  echo "######################################################################"
  echo "###   Full pipeline leave-one-out test: ${test_species} (${NUM_RUNS} runs, ${NUM_GENES} HVG)"
  echo "######################################################################"

  # Checkpoints: fig2/task3_cross_species/leave_one_out_$species/scdiffusion/run$i/{vae,diffusion,classifier}
  ckpt_fold="${CHECKPOINT_ROOT}/fig2/task3_cross_species/leave_one_out_${test_species}/scdiffusion"
  vae_base="${ckpt_fold}/vae_${NUM_GENES}"
  diff_base="${ckpt_fold}/diffusion_${NUM_GENES}"
  cls_base="${ckpt_fold}/classifier_${NUM_GENES}"
  sample_base="samples/fig2/task3_extend/${test_species}_control_ifn/scDiffusion_${NUM_GENES}"
  mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_base}"

  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for leave1out_${test_species}"
    echo "======================"

    vae_dir="${vae_base}/run${i}"
    diff_dir="${diff_base}/run${i}"
    cls_dir="${cls_base}/run${i}"
    run_sample_dir="${sample_base}/run${i}"
    mkdir -p "${vae_dir}" "${diff_dir}/my_diffusion" "${cls_dir}" "${run_sample_dir}"

    vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
    diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
    cls_ckpt="${cls_dir}/model009999.pt"

    echo "--- Step 1: Training VAE ..."
    pushd src/scDiffusion/VAE >/dev/null
    python VAE_train.py \
      --data_dir "${TRAIN_H5}" \
      --num_genes "${NUM_GENES}" \
      --state_dict "${HOMEDIR}/checkpoints/annotation_model_v1" \
      --save_dir "${vae_dir}"
    popd >/dev/null

    echo "--- Step 2: Training Diffusion ..."
    pushd src/scDiffusion >/dev/null
    python cell_train.py \
      --data_dir "${TRAIN_H5}" \
      --vae_path "${vae_ckpt}" \
      --save_dir "${diff_dir}"
    popd >/dev/null

    echo "--- Step 3: Training Classifier ..."
    pushd src/scDiffusion >/dev/null
    python classifier_train.py \
      --data_dir "${TRAIN_H5}" \
      --vae_path "${vae_ckpt}" \
      --model_path "${cls_dir}"
    popd >/dev/null

    echo "--- Step 4: Sampling & Evaluation ..."
    pushd src/scDiffusion >/dev/null
    run_out="$(
      python classifier_sample.py \
        --num_samples "${N_SAMPLES}" \
        --train-data-path "${TRAIN_H5}" \
        --model_path "${diff_ckpt}" \
        --classifier_path "${cls_ckpt}" \
        --ae_dir "${vae_ckpt}" \
        --num_gene "${NUM_GENES}" \
        --sample_dir "${run_sample_dir}" \
        --out_h5ad "${run_sample_dir}/synthetic_ifn_${i}.h5ad" \
        --umap_plot "${run_sample_dir}/umap_comparison_${i}.png" \
        --init_cell_path "${VALID_H5}" 2>&1
    )" || true
    popd >/dev/null

    echo "${run_out}"
    ALL_OUTPUTS+="$(
      printf "%s\n" "${run_out}" | grep -E \
        "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
        || true
    )"
    ALL_OUTPUTS+=$'\n'
  done

  printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="leave1out_${test_species}_control_ifn" -v sp="${test_species}" -v num_runs="${NUM_RUNS}" -v method="scDiffusion(${NUM_GENES})" -v csv_path="${GLOBAL_CSV}" '
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
      return sprintf("%.6f±%.6f", (n?mu:0), (n?sd:0))
    }

    function val(idx, r, v){
      if(idx==1) v=pds[r]; else if(idx==2) v=mae[r]; else if(idx==3) v=des[r]; else if(idx==4) v=edist[r];
      else if(idx==5) v=mmd[r]; else if(idx==6) v=r2[r]; else if(idx==7) v=p_all[r]; else if(idx==8) v=pd_all[r];
      else if(idx==9) v=pd20[r]; else if(idx==10) v=pd50[r]; else if(idx==11) v=pd100[r];
      return (v=="") ? 0 : v;
    }

    END {
      row=ds "," sp "," method;
      for(i=1;i<=11;i++) row=row "," mean_std(i);
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

      print row >> csv_path;
      close(csv_path);
      printf("CSV appended: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished leave1out_${test_species} ---\n"
done

echo "######################################################################"
echo "###   All leave-one-out folds completed!  CSV @ ${GLOBAL_CSV}"
echo "######################################################################"
