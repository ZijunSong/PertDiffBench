#!/usr/bin/env bash
# scDiffusion MOA Diff-MOA split: train & evaluate per MOA with drug+dose conditioning
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
IFS=$'\n\t'
trap 'echo ERROR && exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

# -------------------- Config --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GENES="${NUM_GENES:-3000}"
NUM_RUNS="${NUM_RUNS:-3}"


export WANDB_DISABLED=true
export WANDB_MODE=disabled

# ---------------- Project Root ------------------
HOMEDIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$HOMEDIR"
echo "PWD: $(pwd)"

# ---------------- Paths -------------------------
DATA_BASE="${DATA_BASE:-/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA}"
SAMPLES_BASE="${SAMPLES_BASE:-/data/ppnm/data/PertDiffBench/samples}"
CKPT_BASE="${CKPT_BASE:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"

DATA_ROOT="${DATA_ROOT:-${DATA_BASE}/control_plus_ifn/unseen_diff_moa}"
VAE_STATE_DICT="${VAE_STATE_DICT:-checkpoints/annotation_model_v1}"
SAMPLES_ROOT="${SAMPLES_ROOT:-${SAMPLES_BASE}/fig2/task1_unseenMOA/diff}"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_BASE}/fig2/task1_unseenMOA/diff}"
mkdir -p "${SAMPLES_ROOT}" "${CKPT_ROOT}"

# -------------------- Discover datasets ----------------------
mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_train__plus_control.h5ad" 2>/dev/null | sort)
if [[ ${#TRAIN_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No *_train__plus_control.h5ad found under: ${DATA_ROOT}" >&2
  exit 1
fi

echo "Found ${#TRAIN_FILES[@]} MOA datasets under ${DATA_ROOT}"
echo "Config: runs=${NUM_RUNS} | genes=${NUM_GENES} | samples=${NUM_SAMPLES}"
echo

# -------------------- Main Loop ----------------------
for train_path in "${TRAIN_FILES[@]}"; do
  train_fname="$(basename "${train_path}")"
  moa="${train_fname%_train__plus_control.h5ad}"
  test_fname="${moa}_test__plus_control.h5ad"
  test_path="${DATA_ROOT}/${test_fname}"

  if [[ ! -f "${test_path}" ]]; then
    echo "[WARN] Missing test file for MOA=${moa}: ${test_path}. Skipping." >&2
    continue
  fi

  echo "######################################################################"
  echo "###   scDiffusion MOA: ${moa} (${NUM_RUNS} runs)"
  echo "######################################################################"

  vae_base="${CKPT_ROOT}/scdiffusion/${moa}_${NUM_GENES}/vae"
  diff_base="${CKPT_ROOT}/scdiffusion/${moa}_${NUM_GENES}/diffusion"
  cls_base="${CKPT_ROOT}/scdiffusion/${moa}_${NUM_GENES}/classifier"
  sample_base="${SAMPLES_ROOT}/${moa}/scDiffusion_${NUM_GENES}"
  csv_dir="${sample_base}/metrics"
  mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_base}" "${csv_dir}"

  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for ${moa}"
    echo "======================"

    vae_dir="${vae_base}/run${i}"
    diff_dir="${diff_base}/run${i}"
    cls_dir="${cls_base}/run${i}"
    run_sample_dir="${sample_base}/run${i}"
    mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${run_sample_dir}"

    vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
    diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
    cls_ckpt="${cls_dir}/model009999.pt"
    label_enc_path="${cls_dir}/label_encoder.npz"

    # ---- Step 1: Train VAE ----
    echo "--- Step 1: Training VAE ..."
    if [[ ! -f "${vae_ckpt}" ]]; then
      pushd src/scDiffusion/VAE >/dev/null
      python VAE_train.py \
        --data_dir "../../../${train_path}" \
        --num_genes "${NUM_GENES}" \
        --state_dict "../../../${VAE_STATE_DICT}" \
        --save_dir "../../../${vae_dir}"
      popd >/dev/null
    else
      echo "  [Skip] VAE checkpoint exists: ${vae_ckpt}"
    fi

    # ---- Step 2: Train Diffusion ----
    echo "--- Step 2: Training Diffusion ..."
    if [[ ! -f "${diff_ckpt}" ]]; then
      pushd src/scDiffusion >/dev/null
      python cell_train.py \
        --data_dir "../../${train_path}" \
        --vae_path "../../${vae_ckpt}" \
        --save_dir "../../${diff_dir}"
      popd >/dev/null
    else
      echo "  [Skip] Diffusion checkpoint exists: ${diff_ckpt}"
    fi

    # ---- Step 3: Train Classifier (MOA: drug+dose conditioning) ----
    echo "--- Step 3: Training Classifier (drug+dose) ..."
    if [[ ! -f "${cls_ckpt}" ]]; then
      pushd src/scDiffusion >/dev/null
      python classifier_train.py \
        --data_dir "../../${train_path}" \
        --vae_path "../../${vae_ckpt}" \
        --model_path "../../${cls_dir}" \
        --use_drug_cond \
        --drug_key perturbation \
        --dose_key dose_value
      popd >/dev/null
    else
      echo "  [Skip] Classifier checkpoint exists: ${cls_ckpt}"
    fi

    # ---- Step 4: Sampling & Evaluation (MOA: use drug+dose from test as target) ----
    echo "--- Step 4: Sampling & Evaluation ..."
    pushd src/scDiffusion >/dev/null
    run_out="$(
      python classifier_sample.py \
        --num_samples "${NUM_SAMPLES}" \
        --train-data-path "../../${train_path}" \
        --model_path "../../${diff_ckpt}" \
        --classifier_path "../../${cls_ckpt}" \
        --ae_dir "../../${vae_ckpt}" \
        --num_gene "${NUM_GENES}" \
        --sample_dir "../../${run_sample_dir}" \
        --out_h5ad "../../${run_sample_dir}/synthetic_${moa}_${i}.h5ad" \
        --umap_plot "../../${run_sample_dir}/umap_comparison_${i}.png" \
        --init_cell_path "../../${test_path}" \
        --use_drug_cond \
        --label_encoder_path "../../${label_enc_path}" \
        --drug_key perturbation \
        --dose_key dose_value 2>&1
    )" || true
    popd >/dev/null

    echo "${run_out}"
    ALL_OUTPUTS+="${run_out}"$'\n'
  done

  # ---- Step 5: Aggregate metrics and write CSV ----
  echo
  printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="${moa}_test" -v num_runs="${NUM_RUNS}" -v method="scDiffusion(${NUM_GENES})" -v csv_path="${csv_dir}/metrics_${moa}.csv" '
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
    /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
    /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
    /^E-Distance:/                              { edist[c_edist++] = to_num($NF) }
    /E-Distance:/                               { edist[c_edist++] = to_num($NF) }
    /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = to_num($NF) }
    /R-squared \(R2\):/                         { r2[c_r2++] = to_num($NF) }
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
      printf "==================================================================\n";
      printf " Final statistics for %s (%d runs)\n", ds, num_runs;
      printf "==================================================================\n";
      for(i=1;i<=11;i++) {
        ms = mean_std(i); n=split(ms,a,"±"); if(n>=1) printf "  %d: %s\n", i, ms;
      }
      row=ds "," method;
      for(i=1;i<=11;i++) row=row "," mean_std(i);
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);
      print row >> csv_path;
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished MOA: ${moa} ---\n"
done

echo "######################################################################"
echo "###   All MOAs completed! CSVs under ${SAMPLES_ROOT}/*/scDiffusion_*/metrics/"
echo "######################################################################"

# ---- Step 6: Aggregate all MOA CSVs ----
echo
echo "######################################################################"
echo "###   Aggregating all MOA results into a single CSV file..."
echo "######################################################################"

AGGREGATED_CSV="${SAMPLES_ROOT}/aggregated_metrics_scDiffusion_${NUM_GENES}.csv"
python3 "${HOMEDIR}/utils/aggregate_metrics.py" \
  --samples-root "${SAMPLES_ROOT}" \
  --output-csv "${AGGREGATED_CSV}" \
  --pattern "scDiffusion_${NUM_GENES}"

if [[ -f "${AGGREGATED_CSV}" ]]; then
  echo
  echo "######################################################################"
  echo "###   Aggregation completed!"
  echo "###   Aggregated CSV: ${AGGREGATED_CSV}"
  echo "###   Absolute path: $(cd "$(dirname "${AGGREGATED_CSV}")" && pwd)/$(basename "${AGGREGATED_CSV}")"
  echo "######################################################################"
else
  echo "[WARN] Failed to create aggregated CSV file"
fi
