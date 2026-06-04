#!/usr/bin/env bash
# Fig4 Squidiff — 原文 addition（Δz_sem + DDIM 条件解码）
# 单卡 GPU 0 | H20 大 batch 加速
set -euo pipefail
export PYTHONUNBUFFERED=1
trap 'echo "[ERROR] command failed at line ${LINENO}" >&2' ERR

HOMEDIR="$(cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")" && pwd)"
cd "$HOMEDIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pertdiffbench 2>/dev/null || true
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------- H20 141GB 推荐参数 ----------
NUM_GENES="${NUM_GENES:-3000}"
NUM_RUNS="${NUM_RUNS:-3}"
N_SAMPLES="${N_SAMPLES:-1200}"          # 2h/8h 各 ~1.2k 细胞，尽量用满
BATCH_SIZE="${BATCH_SIZE:-4096}"        # 训练 batch
SAMPLE_BATCH="${SAMPLE_BATCH:-512}"       # DDIM decode batch
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-100000}"
USE_FP16="${USE_FP16:-False}"   # MLPModel 不支持 convert_to_fp16，须保持 False
METHOD_NAME="${METHOD_NAME:-Squidiff-addition}"
SQUIDIFF_METHOD="addition"

LOGDIR="${LOGDIR:-logs}"
DATA_FIG4="${DATA_FIG4:-/data/ppnm/data/PertDiffBench/data/fig4_task1}"
TRAIN_H5="${TRAIN_H5:-${DATA_FIG4}/fig4_train.h5ad}"
TEST_H5="${TEST_H5:-${DATA_FIG4}/fig4_test.h5ad}"

CKPT_BASE="${CKPT_BASE:-checkpoints/fig4/squidiff_addition_${NUM_GENES}}"
SAMPLE_BASE="${SAMPLE_BASE:-samples/fig4/squidiff_addition_${NUM_GENES}}"
CSV_PATH="${SAMPLE_BASE}/metrics_${METHOD_NAME}_fig4.csv"
LOG_FILE="${LOGDIR}/fig4_task1/squidiff_${SQUIDIFF_METHOD}_gpu0.log"

mkdir -p "${CKPT_BASE}" "${SAMPLE_BASE}" "${LOGDIR}/fig4_task1"

{
  echo "== $(date '+%F %T') | fig4 Squidiff-${SQUIDIFF_METHOD} | GPU=${CUDA_VISIBLE_DEVICES} =="
  echo "Config: runs=${NUM_RUNS} genes=${NUM_GENES} n_samples=${N_SAMPLES} train_batch=${BATCH_SIZE} sample_batch=${SAMPLE_BATCH} fp16=${USE_FP16}"
  echo "Ckpt: ${CKPT_BASE} | Samples: ${SAMPLE_BASE}"
  all_outputs=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo "====================== Run ${i}/${NUM_RUNS} ======================"
    run_ckpt="${CKPT_BASE}/run${i}"
    run_sample="${SAMPLE_BASE}/run${i}"
    mkdir -p "${run_ckpt}" "${run_sample}"

    echo "--- Training [run ${i}] ---"
    if ! python src/Squidiff/train_squidiff.py \
      --logger_path "${LOGDIR}/squidiff/fig4_${SQUIDIFF_METHOD}_run${i}" \
      --data_path "${TRAIN_H5}" \
      --resume_checkpoint "${run_ckpt}" \
      --gene_size "${NUM_GENES}" \
      --output_dim "${NUM_GENES}" \
      --batch_size "${BATCH_SIZE}" \
      --lr_anneal_steps "${LR_ANNEAL_STEPS}" \
      --use_fp16 "${USE_FP16}"; then
      echo "[ERROR] Training failed for run ${i}; skip sampling/eval." >&2
      continue
    fi

    if [[ ! -f "${run_ckpt}/model.pt" ]]; then
      echo "[ERROR] Checkpoint missing after training: ${run_ckpt}/model.pt" >&2
      continue
    fi

    echo "--- Sampling 4h/6h (${SQUIDIFF_METHOD}) [run ${i}] ---"
    if ! python scripts/fig4/sample_fig4_squidiff_interp.py \
      --model_path "${run_ckpt}/model.pt" \
      --train-h5ad "${TRAIN_H5}" \
      --out-h5ad "${run_sample}/synthetic_fig4.h5ad" \
      --n-samples "${N_SAMPLES}" \
      --gene-size "${NUM_GENES}" \
      --output-dim "${NUM_GENES}" \
      --batch-size "${SAMPLE_BATCH}" \
      --method "${SQUIDIFF_METHOD}" \
      --anchor-start 2h \
      --anchor-end 8h \
      --target-times 4h 6h; then
      echo "[ERROR] Sampling failed for run ${i}; skip eval." >&2
      continue
    fi

    if [[ ! -f "${run_sample}/synthetic_fig4.h5ad" ]]; then
      echo "[ERROR] Generated h5ad missing: ${run_sample}/synthetic_fig4.h5ad" >&2
      continue
    fi

    echo "--- Eval [run ${i}] ---"
    output=$(python scripts/fig4/eval_fig4_time_conditioned.py \
      --test-h5ad "${TEST_H5}" \
      --generated-h5ad "${run_sample}/synthetic_fig4.h5ad" \
      --train-h5ad "${TRAIN_H5}" \
      --n-samples "${N_SAMPLES}" \
      --method-name "${METHOD_NAME}" 2>&1) || true
    echo "$output"
    all_outputs+="$output\n"
  done

  echo -e "$all_outputs" | awk -v dataset="fig4" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
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
    function mean_std(idx,    i,n,s,mu,ss,v) {
      if (idx==1)  { n=c_pds; for(i=0;i<n;i++){v=pds[i]; s+=v} }
      else if(idx==2){ n=c_mae; for(i=0;i<n;i++){v=mae[i]; s+=v} }
      else if(idx==3){ n=c_des; for(i=0;i<n;i++){v=des[i]; s+=v} }
      else if(idx==4){ n=c_edist; for(i=0;i<n;i++){v=edist[i]; s+=v} }
      else if(idx==5){ n=c_mmd; for(i=0;i<n;i++){v=mmd[i]; s+=v} }
      else if(idx==6){ n=c_r2; for(i=0;i<n;i++){v=r2[i]; s+=v} }
      else if(idx==7){ n=c_pearson_all; for(i=0;i<n;i++){v=pearson_all[i]; s+=v} }
      else if(idx==8){ n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i]; s+=v} }
      else if(idx==9){ n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i]; s+=v} }
      else if(idx==10){ n=c_pearson_delta_de50; for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v} }
      else if(idx==11){ n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
      mu = (n>0)? s/n : 0;
      for(i=0;i<n;i++){
        if (idx==1) v=pds[i]; else if(idx==2) v=mae[i]; else if(idx==3) v=des[i]; else if(idx==4) v=edist[i];
        else if(idx==5) v=mmd[i]; else if(idx==6) v=r2[i]; else if(idx==7) v=pearson_all[i];
        else if(idx==8) v=pearson_delta_all[i]; else if(idx==9) v=pearson_delta_de20[i];
        else if(idx==10) v=pearson_delta_de50[i]; else if(idx==11) v=pearson_delta_de100[i];
        ss += (v - mu) * (v - mu);
      }
      return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
    }
    function val(idx, j,    v){
      if (idx==1) v=pds[j]; else if(idx==2) v=mae[j]; else if(idx==3) v=des[j]; else if(idx==4) v=edist[j];
      else if(idx==5) v=mmd[j]; else if(idx==6) v=r2[j]; else if(idx==7) v=pearson_all[j];
      else if(idx==8) v=pearson_delta_all[j]; else if(idx==9) v=pearson_delta_de20[j];
      else if(idx==10) v=pearson_delta_de50[j]; else if(idx==11) v=pearson_delta_de100[j];
      return v;
    }
    END {
      print "==================================================================";
      printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
      print "==================================================================";
      metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES"; metric_names[4]="E-Distance";
      metric_names[5]="MMD"; metric_names[6]="R2"; metric_names[7]="Pearson (all genes)";
      metric_names[8]="Pearson Delta (all genes)"; metric_names[9]="Pearson Delta (top 20 DE genes)";
      metric_names[10]="Pearson Delta (top 50 DE genes)"; metric_names[11]="Pearson Delta (top 100 DE genes)";
      for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); printf "%-40s: %.4f ± %.4f\n", metric_names[i], parts[1], parts[2]; }
      header = "Method"; for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
      for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];
      row = method; for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
      for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));
      print header > csv_path; print row >> csv_path; close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '
  echo "--- Finished fig4 Squidiff-${SQUIDIFF_METHOD} on GPU ${CUDA_VISIBLE_DEVICES} ---"
} 2>&1 | tee -a "${LOG_FILE}"

echo "######################################################################"
echo "###   fig4 Squidiff-addition (GPU0) complete.                       ###"
echo "######################################################################"
