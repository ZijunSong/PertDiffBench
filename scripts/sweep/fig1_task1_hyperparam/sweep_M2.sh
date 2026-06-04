#!/bin/bash
# Fig2 task2+ unseen cell type hyperparam sweep — DDPM+MLP M2 (higher lr)
# LOO holdout CD4T @ p0.25 | 6998 genes, 1 run | lr=1e-5, batch_size=1024 (H20)
set -e
trap 'echo "ERROR: sweep M2 failed." >&2' ERR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

HOMEDIR="/data/ppnm/PertDiffBench"
ROOT_DIR="/data/ppnm/data/PertDiffBench/"
CKPT_ROOT="/data/ppnm/checkpoints/PertDiffBench/checkpoints"
cd "${HOMEDIR}"
export PYTHONPATH="${HOMEDIR}:${PYTHONPATH:-}"

SWEEP_ID="M2"
HOLDOUT="CD4T"
CTRL_SLUG="${CTRL_SLUG:-p0.25}"
NUM_GENES=6998
N_SAMPLES=256
CONFIG_FILE="configs/baselines/sweep/fig2_unseen_cd4t/mlp_ddpm_mlp_${SWEEP_ID}.yaml"
CKPT_NAME="model_epoch_1000.pth"
METHOD_NAME="DDPM+MLP-${SWEEP_ID}(unseenCD4T_${CTRL_SLUG},lr=1e-5,bs=1024)"

DATA_DIR="/data/ppnm/PertDiffBench/data/fig2/task2_unseen_celltype_plus/loo_${HOLDOUT}/${CTRL_SLUG}"
train_data="${DATA_DIR}/scgen_combined_train_plus_test_control.h5ad"
valid_data="${DATA_DIR}/task2_test_exp.h5ad"
save_dir="${CKPT_ROOT}/sweep/fig2_unseen_cd4t/${SWEEP_ID}/run1"
sample_dir="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${SWEEP_ID}/run1"
csv_path="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${SWEEP_ID}/metrics_${SWEEP_ID}_${HOLDOUT}_${CTRL_SLUG}.csv"
log_dir="${HOMEDIR}/logs/sweep/fig2_unseen_cd4t"
mkdir -p "${save_dir}" "${sample_dir}" "${log_dir}"

{
  echo "== $(date '+%F %T') | sweep=${SWEEP_ID} | holdout=${HOLDOUT} ${CTRL_SLUG} | GPU=${CUDA_VISIBLE_DEVICES} | lr=1e-5 bs=1024 =="

  python scripts/baseline_exp/train_mlp_ddpm_mlp.py \
    --config "${CONFIG_FILE}" \
    --data-path "${train_data}" \
    --save-weight-dir "${save_dir}" \
    --gene-nums "${NUM_GENES}" \
    --pair-only-obs-key "split" \
    --pair-only-obs-value "train"

  output=$(
    python scripts/baseline_exp/eval_mlp_ddpm_mlp.py \
      --config "${CONFIG_FILE}" \
      --train-data-path "${train_data}" \
      --data-path "${valid_data}" \
      --ckpt "${save_dir}/${CKPT_NAME}" \
      --out_h5ad "${sample_dir}/synthetic_ifn.h5ad" \
      --n_samples "${N_SAMPLES}" \
      --gene-nums "${NUM_GENES}" 2>&1
  ) || true
  echo "$output"

  echo "$output" | awk -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
    /Perturbation Discrimination Score \(PDS\):/ { pds=$NF }
    /Mean Absolute Error \(MAE\):/              { mae=$NF }
    /Differential Expression Score \(DES\):/    { des=$NF }
    /E-Distance:/                               { edist=$NF }
    /Maximum Mean Discrepancy \(MMD\):/         { mmd=$NF }
    /R-squared \(R2\):/                         { r2=$NF }
    /Pearson \(all genes\):/                    { pa=$NF }
    /Pearson Delta \(all genes\):/              { pda=$NF }
    /Pearson Delta \(top 20 DE genes\):/        { pd20=$NF }
    /Pearson Delta \(top 50 DE genes\):/        { pd50=$NF }
    /Pearson Delta \(top 100 DE genes\):/       { pd100=$NF }
    END {
      hdr="Method,PDS,MAE,DES,E-Distance,MMD,R2,Pearson_all,PearsonDelta_all,PearsonDelta_DE20,PearsonDelta_DE50,PearsonDelta_DE100";
      row=sprintf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", method,pds,mae,des,edist,mmd,r2,pa,pda,pd20,pd50,pd100);
      print hdr > csv_path; print row >> csv_path; close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '
  echo "--- sweep ${SWEEP_ID} done ---"
} 2>&1 | tee "${log_dir}/${SWEEP_ID}.log"
