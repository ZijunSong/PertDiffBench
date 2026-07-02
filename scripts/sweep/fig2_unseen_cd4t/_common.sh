#!/bin/bash
# Shared runner for Fig2 task2+ unseen-CD4T hyperparameter sweeps.
# Each wrapper sets: SWEEP_METHOD, SWEEP_AXIS, SWEEP_TAG, and axis-specific values.
# DDPM and DDPM+MLP share the same default hyperparameters and sweep grids:
#   lr: 2e-5 / 1e-5 / 5e-6
#   bs: 1024 / 2048 / 4096
#   steps: 500 / 1000 / 2000
#   beta_T (beta_1=1e-4): 0.01 / 0.02 / 0.04
set -euo pipefail

: "${SWEEP_METHOD:?Set SWEEP_METHOD to ddpm or ddpm_mlp}"
: "${SWEEP_AXIS:?Set SWEEP_AXIS to lr, bs, steps, or beta}"
: "${SWEEP_TAG:?Set SWEEP_TAG to a filesystem-safe value tag}"

HOMEDIR="${HOMEDIR:-/data/ppnm/PertDiffBench}"
ROOT_DIR="${ROOT_DIR:-/data/ppnm/data/PertDiffBench/}"
CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"
HOLDOUT="${HOLDOUT:-CD4T}"
CTRL_SLUG="${CTRL_SLUG:-p0.25}"
NUM_GENES="${NUM_GENES:-6998}"
source "${HOMEDIR}/scripts/lib/max_n_samples.sh"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"

cd "${HOMEDIR}"
export PYTHONPATH="${HOMEDIR}:${PYTHONPATH:-}"

# Shared hyperparameter defaults for fair DDPM vs DDPM+MLP comparison.
DEFAULT_LR="1e-5"
DEFAULT_BS="2048"
DEFAULT_STEPS="1000"
DEFAULT_BETA1="1e-4"
DEFAULT_BETAT="0.02"

if [[ "${SWEEP_METHOD}" == "ddpm" ]]; then
  TRAIN_SCRIPT="scripts/baseline_exp/train_scrna_ddpm_scrna.py"
  EVAL_SCRIPT="scripts/baseline_exp/eval_scrna_ddpm_scrna.py"
  CKPT_NAME="scrna_ddpm_epoch${TRAIN_EPOCHS}.pt"
  METHOD_LABEL="DDPM"
elif [[ "${SWEEP_METHOD}" == "ddpm_mlp" ]]; then
  TRAIN_SCRIPT="scripts/baseline_exp/train_mlp_ddpm_mlp.py"
  EVAL_SCRIPT="scripts/baseline_exp/eval_mlp_ddpm_mlp.py"
  CKPT_NAME="model_epoch_${TRAIN_EPOCHS}.pth"
  METHOD_LABEL="DDPM+MLP"
else
  echo "Unknown SWEEP_METHOD=${SWEEP_METHOD}" >&2
  exit 1
fi

SWEEP_LR="${SWEEP_LR:-${DEFAULT_LR}}"
SWEEP_BS="${SWEEP_BS:-${DEFAULT_BS}}"
SWEEP_STEPS="${SWEEP_STEPS:-${DEFAULT_STEPS}}"
SWEEP_BETA1="${SWEEP_BETA1:-${DEFAULT_BETA1}}"
SWEEP_BETAT="${SWEEP_BETAT:-${DEFAULT_BETAT}}"

RUN_ID="${SWEEP_METHOD}__${SWEEP_AXIS}__${SWEEP_TAG}"
DATA_DIR="${HOMEDIR}/data/fig2/task2_unseen_celltype_plus/loo_${HOLDOUT}/${CTRL_SLUG}"
train_data="${DATA_DIR}/scgen_combined_train_plus_test_control.h5ad"
valid_data="${DATA_DIR}/task2_test_exp.h5ad"
N_SAMPLES="$(max_n_samples_multi_pert "${valid_data}")"

save_dir="${CKPT_ROOT}/sweep/fig2_unseen_cd4t/${RUN_ID}/run1"
sample_dir="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${RUN_ID}/run1"
csv_path="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${RUN_ID}/metrics_${HOLDOUT}_${CTRL_SLUG}.csv"
log_dir="${HOMEDIR}/logs/sweep/fig2_unseen_cd4t/${SWEEP_AXIS}"
config_dir="${HOMEDIR}/configs/baselines/sweep/fig2_unseen_cd4t/generated"
CONFIG_FILE="${config_dir}/${RUN_ID}.yaml"

mkdir -p "${save_dir}" "${sample_dir}" "${log_dir}" "${config_dir}"

if [[ "${SWEEP_METHOD}" == "ddpm" ]]; then
  cat > "${CONFIG_FILE}" <<EOF
model:
  input_dim: ${NUM_GENES}
  hidden_dim: 1024

diffusion:
  beta_1: ${SWEEP_BETA1}
  beta_T: ${SWEEP_BETAT}
  timesteps: ${SWEEP_STEPS}

data:
  path: ./dataset/scrna_data/scrna.h5ad
  batch_size: 32
  label_key: perturbation_status

train:
  device: cuda
  epoch: ${TRAIN_EPOCHS}
  batch_size: ${SWEEP_BS}
  lr: ${SWEEP_LR}
  weight_decay: 1.0e-4
  grad_clip: 1.0
  grad_clip_norm: 1.0
  warmup_multiplier: 10
  save_weight_dir: ${save_dir}
  resume_from: null
  num_workers: 32
  ckpt_save_interval: 100

sample:
  batch_size: 64
  sampled_dir: ${sample_dir}
  out_h5ad: synthetic_${RUN_ID}.h5ad
EOF
else
  cat > "${CONFIG_FILE}" <<EOF
model:
  ae:
    input_dim: ${NUM_GENES}
    latent_dim: 256
    hidden_dim: 1024
  diffusion:
    timesteps: ${SWEEP_STEPS}
    beta_1: ${SWEEP_BETA1}
    beta_T: ${SWEEP_BETAT}
    hidden_dim: 512

data:
  path: ./dataset/scrna_data/scrna.h5ad
  batch_size: 32
  label_key: Condition
  num_workers: 4

train:
  batch_size: ${SWEEP_BS}
  device: cuda
  epoch: ${TRAIN_EPOCHS}
  lr: ${SWEEP_LR}
  weight_decay: 1.0e-4
  grad_clip: 0.5
  warmup_multiplier: 10
  save_weight_dir: ${save_dir}
  resume_from: null
  num_workers: 32
  ckpt_save_interval: 500

sample:
  batch_size: 64
  sampled_dir: ${sample_dir}
  out_h5ad: synthetic_${RUN_ID}.h5ad
EOF
fi

METRIC_METHOD="${METHOD_LABEL}(${RUN_ID}|lr=${SWEEP_LR}|bs=${SWEEP_BS}|T=${SWEEP_STEPS}|beta=${SWEEP_BETA1}-${SWEEP_BETAT})"

{
  echo "== $(date '+%F %T') | ${RUN_ID} | holdout=${HOLDOUT} ${CTRL_SLUG} | GPU=${CUDA_VISIBLE_DEVICES:-unset} =="
  echo "Config: ${CONFIG_FILE}"

  python "${TRAIN_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --data-path "${train_data}" \
    --save-weight-dir "${save_dir}" \
    --gene-nums "${NUM_GENES}" \
    --pair-only-obs-key "split" \
    --pair-only-obs-value "train"

  output=$(
    python "${EVAL_SCRIPT}" \
      --config "${CONFIG_FILE}" \
      --train-data-path "${train_data}" \
      --data-path "${valid_data}" \
      --ckpt "${save_dir}/${CKPT_NAME}" \
      --out_h5ad "${sample_dir}/synthetic_ifn.h5ad" \
      --n_samples "${N_SAMPLES}" \
      --gene-nums "${NUM_GENES}" 2>&1
  ) || true
  echo "${output}"

  echo "${output}" | awk -v method="${METRIC_METHOD}" -v csv_path="${csv_path}" '
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
  echo "--- ${RUN_ID} done ---"
} 2>&1 | tee "${log_dir}/${RUN_ID}.log"
