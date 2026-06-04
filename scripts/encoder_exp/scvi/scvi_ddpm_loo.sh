#!/usr/bin/env bash
# scVI + latent DDPM on Fig2 task2+ (leave-one-out cell type × control fraction p0 / p0.25 / p0.5).
# Saves encoder input latents (train & test) as .npy under each fold directory.
#
# Default GPU 0 (override CUDA_VISIBLE_DEVICES). From repo root:
#   bash scripts/encoder_exp/scvi/scvi_ddpm_loo.sh
#
# Optional: FIG2_TASK2_LOO_HOLDOUT_TYPES="B NK"  FIG2_TASK2_LOO_CTRL_SLUGS="p0"
set -euo pipefail
IFS=$'\n\t'

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

NUM_RUNS="${NUM_RUNS:-3}"
N_SAMPLES="${N_SAMPLES:-256}"
METHOD_NAME="${METHOD_NAME:-scvi_ddpm}"
CONFIG_PATH="${CONFIG_PATH:-configs/baselines/scvi_ddpm_mlp.yaml}"
DATA_BASE="${DATA_BASE:-data/fig2/task2_unseen_celltype_plus}"

HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"
DATA_ROOT="${HOMEDIR}/${DATA_BASE}"

if [[ -n "${FIG2_TASK2_LOO_HOLDOUT_TYPES:-}" ]]; then
  read -r -a HOLDOUT_TYPES <<< "${FIG2_TASK2_LOO_HOLDOUT_TYPES}"
else
  HOLDOUT_TYPES=( "B" "CD4T" "CD8T" "CD14+Mono" "Dendritic" "FCGR3A+Mono" "NK" )
fi
if [[ -n "${FIG2_TASK2_LOO_CTRL_SLUGS:-}" ]]; then
  read -r -a CTRL_SLUGS <<< "${FIG2_TASK2_LOO_CTRL_SLUGS}"
else
  CTRL_SLUGS=( "p0" "p0.25" "p0.5" )
fi

OUT_ROOT="samples/encoder_exp/fig2_task2_loo/${METHOD_NAME}"
CKPT_ROOT="checkpoints/encoder_exp/fig2_task2_loo/${METHOD_NAME}"
GLOBAL_CSV="${OUT_ROOT}/metrics_all.csv"
LOG_DIR_ROOT="logs/encoder_exp/fig2_task2_loo/${METHOD_NAME}"

mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}" "${LOG_DIR_ROOT}"

if [[ ! -f "${GLOBAL_CSV}" ]]; then
  {
    printf "Dataset,Method"
    printf ",PDS (mean±std),MAE (mean±std),DES (mean±std),E-Distance (mean±std),MMD (mean±std),R2 (mean±std)"
    printf ",Pearson (all genes) (mean±std),Pearson Delta (all genes) (mean±std)"
    printf ",Pearson Delta (top 20 DE genes) (mean±std),Pearson Delta (top 50 DE genes) (mean±std),Pearson Delta (top 100 DE genes) (mean±std)"
    for r in 1 2 3; do
      printf ",Run%d PDS,Run%d MAE,Run%d DES,Run%d E-Distance,Run%d MMD,Run%d R2" "$r" "$r" "$r" "$r" "$r" "$r"
      printf ",Run%d Pearson (all genes),Run%d Pearson Delta (all genes)" "$r" "$r"
      printf ",Run%d Pearson Delta (top 20 DE genes),Run%d Pearson Delta (top 50 DE genes),Run%d Pearson Delta (top 100 DE genes)" "$r" "$r" "$r"
    done
    printf "\n"
  } > "${GLOBAL_CSV}"
fi

for ht in "${HOLDOUT_TYPES[@]}"; do
  for slug in "${CTRL_SLUGS[@]}"; do
    ds_tag="${ht}_${slug}"
    DATA_DIR="${DATA_ROOT}/loo_${ht}/${slug}"
    TRAIN_H5="${DATA_DIR}/task2_train_exp.h5ad"
    VALID_H5="${DATA_DIR}/task2_test_exp.h5ad"
    FOLD_OUT="${OUT_ROOT}/${ht}/${slug}"

    if [[ ! -f "${TRAIN_H5}" || ! -f "${VALID_H5}" ]]; then
      echo "[WARN] Skip ${ds_tag}: missing data under ${DATA_DIR}"
      continue
    fi

    mkdir -p "${FOLD_OUT}"
    SCVI_MODEL_DIR="${FOLD_OUT}/scvi_encoder"
    SCVI_OUT_H5="${FOLD_OUT}/task2_train_with_scvi_latent.h5ad"
    VALID_WITH_LATENT="${FOLD_OUT}/task2_test_with_scvi_latent.h5ad"

    echo "######################################################################"
    echo "###   ${METHOD_NAME} LOO | ${ds_tag}"
    echo "######################################################################"

    mkdir -p "${SCVI_MODEL_DIR}"

    echo "--- [1/3] Train scVI on train split (${ds_tag}) ---"
    python scripts/encoder_exp/scvi/train_scvi_and_latent_ddpm.py \
      --data-path "${TRAIN_H5}" \
      --out-h5ad "${SCVI_OUT_H5}" \
      --model-dir "${SCVI_MODEL_DIR}" \
      --n-latent 32 \
      --max-epochs 1000 \
      --gpu 2>&1 | tee "${LOG_DIR_ROOT}/train_scvi_${ds_tag}.log"

    echo "--- [2/3] Encode test h5ad with scVI ---"
    python scripts/encoder_exp/scvi/apply_scvi_encoder.py \
      --data-path "${VALID_H5}" \
      --out-h5ad "${VALID_WITH_LATENT}" \
      --model-dir "${SCVI_MODEL_DIR}" \
      --gpu 2>&1 | tee "${LOG_DIR_ROOT}/encode_test_scvi_${ds_tag}.log"

    python scripts/encoder_exp/save_encoder_input_latent.py \
      --h5ad "${SCVI_OUT_H5}" \
      --latent-key "X_scvi" \
      --out-npy "${FOLD_OUT}/encoder_input_train_latent.npy" \
      --out-obs-names "${FOLD_OUT}/encoder_input_train_obs_names.txt"

    python scripts/encoder_exp/save_encoder_input_latent.py \
      --h5ad "${VALID_WITH_LATENT}" \
      --latent-key "X_scvi" \
      --out-npy "${FOLD_OUT}/encoder_input_test_latent.npy" \
      --out-obs-names "${FOLD_OUT}/encoder_input_test_obs_names.txt"

    CELL_BUF="${FOLD_OUT}/_agg_buffer.txt"
    : > "${CELL_BUF}"

    for (( run=1; run<=NUM_RUNS; run++ )); do
      RUN_CKPT_DIR="${CKPT_ROOT}/${ht}/${slug}/run_${run}"
      mkdir -p "${RUN_CKPT_DIR}"

      echo "--- [3/3] Run ${run}/${NUM_RUNS}: train DDPM + eval ---"
      python scripts/encoder_exp/scvi/train_scvi_latent_ddpm_mlp.py \
        -c "${CONFIG_PATH}" \
        --train-data-path "${SCVI_OUT_H5}" \
        --save-weight-dir "${RUN_CKPT_DIR}" \
        2>&1 | tee "${LOG_DIR_ROOT}/train_ddpm_${ds_tag}_run${run}.log"

      CKPT_PATH="${RUN_CKPT_DIR}/model_final.pth"
      run_output="$(python scripts/encoder_exp/scvi/eval_scvi_latent_ddpm_mlp.py \
        -c "${CONFIG_PATH}" \
        -k "${CKPT_PATH}" \
        --data-path "${VALID_WITH_LATENT}" \
        -n "${N_SAMPLES}" \
        -o "${FOLD_OUT}/preds_run_${run}.h5ad" 2>&1)" || true
      echo "${run_output}"

      {
        printf "%s\n" "${run_output}" | grep -E \
          "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
          || true
        printf "\n"
      } >> "${CELL_BUF}"
    done

    awk -v ds="${ds_tag}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${GLOBAL_CSV}" '
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
        if(idx==1) v=pds[r]; else if(idx==2) v=mae[r]; else if(idx==3) v=des[r]; else if(idx==4) v=edist[r];
        else if(idx==5) v=mmd[r]; else if(idx==6) v=r2[r]; else if(idx==7) v=p_all[r]; else if(idx==8) v=pd_all[r];
        else if(idx==9) v=pd20[r]; else if(idx==10) v=pd50[r]; else if(idx==11) v=pd100[r];
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
    ' "${CELL_BUF}"

  done
done

echo "### ${METHOD_NAME} Fig2 task2 LOO done. CSV => ${GLOBAL_CSV}"
