#!/bin/bash
# Re-run DDPM+MLP eval only (checkpoints already trained) and merge all sweep CSVs.
set -euo pipefail

HOMEDIR="${HOMEDIR:-/data/ppnm/PertDiffBench}"
ROOT_DIR="${ROOT_DIR:-/data/ppnm/data/PertDiffBench/}"
CKPT_ROOT="${CKPT_ROOT:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"
HOLDOUT="${HOLDOUT:-CD4T}"
CTRL_SLUG="${CTRL_SLUG:-p0.25}"
NUM_GENES="${NUM_GENES:-6998}"
N_SAMPLES="${N_SAMPLES:-256}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"

cd "${HOMEDIR}"
export PYTHONPATH="${HOMEDIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

DATA_DIR="${HOMEDIR}/data/fig2/task2_unseen_celltype_plus/loo_${HOLDOUT}/${CTRL_SLUG}"
train_data="${DATA_DIR}/scgen_combined_train_plus_test_control.h5ad"
valid_data="${DATA_DIR}/task2_test_exp.h5ad"
EVAL_SCRIPT="scripts/baseline_exp/eval_mlp_ddpm_mlp.py"
CONFIG_DIR="${HOMEDIR}/configs/baselines/sweep/fig2_unseen_cd4t/generated"

run_ids=(
  ddpm_mlp__lr__5e-6 ddpm_mlp__lr__1e-5 ddpm_mlp__lr__2e-5
  ddpm_mlp__bs__1024 ddpm_mlp__bs__2048 ddpm_mlp__bs__4096
  ddpm_mlp__steps__500 ddpm_mlp__steps__1000 ddpm_mlp__steps__2000
  ddpm_mlp__beta__1e-4_0.01 ddpm_mlp__beta__1e-4_0.02 ddpm_mlp__beta__1e-4_0.04
)

for run_id in "${run_ids[@]}"; do
  save_dir="${CKPT_ROOT}/sweep/fig2_unseen_cd4t/${run_id}/run1"
  sample_dir="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${run_id}/run1"
  csv_path="${ROOT_DIR}samples/sweep/fig2_unseen_cd4t/${run_id}/metrics_${HOLDOUT}_${CTRL_SLUG}.csv"
  config_file="${CONFIG_DIR}/${run_id}.yaml"
  ckpt="${save_dir}/model_epoch_1000.pth"

  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP ${run_id}: missing checkpoint ${ckpt}" >&2
    continue
  fi

  axis="${run_id#ddpm_mlp__}"
  axis="${axis%%__*}"
  log_dir="${HOMEDIR}/logs/sweep/fig2_unseen_cd4t/${axis}"
  mkdir -p "${sample_dir}" "${log_dir}"

  lr=$(awk '/^train:/{flag=1;next} /^[^ ]/ {flag=0} flag && /^  lr:/ {print $2; exit}' "${config_file}")
  bs=$(awk '/^train:/{flag=1;next} /^[^ ]/ {flag=0} flag && /^  batch_size:/ {print $2; exit}' "${config_file}")
  steps=$(awk '/timesteps:/ {print $2; exit}' "${config_file}")
  beta1=$(awk '/beta_1:/ {print $2; exit}' "${config_file}")
  betat=$(awk '/beta_T:/ {print $2; exit}' "${config_file}")
  metric_method="DDPM+MLP(${run_id}|lr=${lr}|bs=${bs}|T=${steps}|beta=${beta1}-${betat})"

  echo "== Re-eval ${run_id} =="
  output=$(
    python "${EVAL_SCRIPT}" \
      --config "${config_file}" \
      --train-data-path "${train_data}" \
      --data-path "${valid_data}" \
      --ckpt "${ckpt}" \
      --out_h5ad "${sample_dir}/synthetic_ifn.h5ad" \
      --n_samples "${N_SAMPLES}" \
      --gene-nums "${NUM_GENES}" 2>&1
  ) || true
  echo "${output}" | tee "${log_dir}/${run_id}.reeval.log"

  echo "${output}" | awk -v method="${metric_method}" -v csv_path="${csv_path}" '
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
done

python3 "${HOMEDIR}/scripts/sweep/fig2_unseen_cd4t/merge_sweep_metrics.py"
