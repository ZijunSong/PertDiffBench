#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
IFS=$'\n\t'
trap 'echo "[ERROR] command failed" >&2; exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

# 保证从项目根目录运行（nohup 可能从其他目录启动）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# =================== Config ===================
NUM_RUNS="${NUM_RUNS:-3}"
GENE_SIZE="${GENE_SIZE:-3000}"
OUTPUT_DIM="${OUTPUT_DIM:-3000}"
N_SAMPLES="${N_SAMPLES:-100}"
METHOD_NAME="${METHOD_NAME:-Squidiff}"

DATA_ROOT="${DATA_ROOT:-data/fig2/task1_unseenMOA/control_plus_ifn_with_smiles/control_plus_ifn/unseen_diff_moa}"

# 统一的control数据路径
CONTROL_DATA_PATH="data/fig2/task1_unseenMOA/control_merged.h5ad"
[[ -f "${CONTROL_DATA_PATH}" ]] || die "Control file not found: ${CONTROL_DATA_PATH}"

LOGROOT="${LOGROOT:-logs/squidiff}"
OUT_BASE="${OUT_BASE:-samples/fig2/task1_unseenMOA/diff/Squidiff_${GENE_SIZE}_withSMILES}"
CKPT_BASE="${CKPT_BASE:-checkpoints/squidiff/fig2_task1_unseenMOA/diff/${GENE_SIZE}_withSMILES}"
CSV_BASE="${CSV_BASE:-${OUT_BASE}/metrics}"

mkdir -p "${OUT_BASE}" "${CKPT_BASE}" "${CSV_BASE}"

die() { echo "[ERROR] $*" >&2; exit 1; }

# =================== Discover datasets ===================
mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_train__plus_control.h5ad" | sort)
[[ ${#TRAIN_FILES[@]} -gt 0 ]] || die "No *_train__plus_control.h5ad found under: ${DATA_ROOT}"

echo "Found ${#TRAIN_FILES[@]} MOA datasets under ${DATA_ROOT}"
echo "Using unified control data: ${CONTROL_DATA_PATH}"
echo "Config: runs=${NUM_RUNS} | genes=${GENE_SIZE} | output_dim=${OUTPUT_DIM} | n_samples=${N_SAMPLES}"
echo

# ========================= Main Loop =========================
for train_path in "${TRAIN_FILES[@]}"; do
  train_file="$(basename "${train_path}")"
  moa="${train_file%_train__plus_control.h5ad}"
  test_path="${DATA_ROOT}/${moa}_test__plus_control.h5ad"

  [[ -f "${test_path}" ]] || die "Missing test file for MOA=${moa}: ${test_path}"

  echo "######################################################################"
  echo "###   Squidiff for MOA: ${moa} (${NUM_RUNS} runs)"
  echo "######################################################################"

  OUT_ROOT="${OUT_BASE}/${moa}"
  CKPT_ROOT="${CKPT_BASE}/${moa}"
  METRICS_CSV="${CSV_BASE}/metrics_${moa}.csv"

  mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"

  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo
    echo "======================"
    echo " Run ${i}/${NUM_RUNS} for ${moa}"
    echo "======================"

    RUN_CKPT_DIR="${CKPT_ROOT}/run${i}"
    RUN_OUT_DIR="${OUT_ROOT}/run${i}"
    mkdir -p "${RUN_CKPT_DIR}" "${RUN_OUT_DIR}"

    # ---- Step 1: Train（启用SMILES + 统一control数据）----
    echo -e "\n--- Training model for ${moa} (run ${i}) ---"
    python src/Squidiff/train_squidiff.py \
      --logger_path "${LOGROOT}/fig2_task1_unseenMOA_${moa}_run${i}" \
      --data_path "${train_path}" \
      --resume_checkpoint "${RUN_CKPT_DIR}" \
      --gene_size "${GENE_SIZE}" \
      --output_dim "${OUTPUT_DIM}" \
      --use_drug_structure True

    echo "--- Training for ${moa} (run ${i}) complete. ---"

    # ---- Step 2: Evaluate ----
    echo -e "\n--- Evaluating (sampling) for ${moa} (run ${i}) ---"

    PRED_H5AD="${RUN_OUT_DIR}/synthetic_${moa}_run_${i}.h5ad"
    UMAP_PNG="${RUN_OUT_DIR}/umap_comparison_${i}.png"
    MODEL_PT="${RUN_CKPT_DIR}/model.pt"

    # 暂时关闭 ERR trap 和 set -e，否则 Python 非零退出会直接触发 trap 且看不到报错
    set +e
    trap - ERR
    output="$(
      python src/Squidiff/sample_squidiff.py \
        --model_path "${MODEL_PT}" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${OUTPUT_DIM}" \
        --out_h5ad "${PRED_H5AD}" \
        --n_samples "${N_SAMPLES}" \
        --umap_plot "${UMAP_PNG}" \
        --train_data_path "${train_path}" \
        --data_path "${test_path}" \
        --control_data_path "${CONTROL_DATA_PATH}" \
        --use_drug_structure 2>&1 | { if [ -t 1 ]; then tee /dev/tty; else cat; fi; }
    )"
    eval_ret=$?
    trap 'echo "[ERROR] command failed" >&2; exit 1' ERR
    set -e

    # 将 Python 输出写入当前标准输出（nohup 会写入日志），便于排查
    echo "${output}"

    if [ "${eval_ret}" -ne 0 ]; then
      echo "[ERROR] sample_squidiff.py exited with ${eval_ret} for ${moa} run ${i}. See output above for traceback." >&2
      exit 1
    fi

    ALL_OUTPUTS+="${output}"$'\n'
  done

  # ---- Step 3: 统计 + CSV 输出 ----
  echo
  printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="${moa}_test" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${METRICS_CSV}" '
    function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }
    /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
    /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
    /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
    /^E-Distance:/  { edist[c_edist++] = to_num($NF) }
    /Maximum Mean Discrepancy \(MMD\):/  { mmd[c_mmd++] = to_num($NF) }
    /R-squared \(R2\):/  { r2[c_r2++] = to_num($NF) }
    /Pearson \(all genes\):/  { p_all[c_p_all++] = to_num($NF) }
    /Pearson Delta \(all genes\):/  { pd_all[c_pd_all++] = to_num($NF) }
    /Pearson Delta \(top 20 DE genes\):/  { pd20[c_pd20++] = to_num($NF) }
    /Pearson Delta \(top 50 DE genes\):/  { pd50[c_pd50++] = to_num($NF) }
    /Pearson Delta \(top 100 DE genes\):/  { pd100[c_pd100++] = to_num($NF) }
    function mean(a,n,s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n?s/n:0 }
    function std(a,n,mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)^2; return sqrt(s/(n-1)) }
    END {
      row=ds","method
      row=row","sprintf("%.6f±%.6f", mean(pds,c_pds), std(pds,c_pds))
      row=row","sprintf("%.6f±%.6f", mean(mae,c_mae), std(mae,c_mae))
      row=row","sprintf("%.6f±%.6f", mean(des,c_des), std(des,c_des))
      row=row","sprintf("%.6f±%.6f", mean(edist,c_edist), std(edist,c_edist))
      row=row","sprintf("%.6f±%.6f", mean(mmd,c_mmd), std(mmd,c_mmd))
      row=row","sprintf("%.6f±%.6f", mean(r2,c_r2), std(r2,c_r2))
      row=row","sprintf("%.6f±%.6f", mean(p_all,c_p_all), std(p_all,c_p_all))
      row=row","sprintf("%.6f±%.6f", mean(pd_all,c_pd_all), std(pd_all,c_pd_all))
      row=row","sprintf("%.6f±%.6f", mean(pd20,c_pd20), std(pd20,c_pd20))
      row=row","sprintf("%.6f±%.6f", mean(pd50,c_pd50), std(pd50,c_pd50))
      row=row","sprintf("%.6f±%.6f", mean(pd100,c_pd100), std(pd100,c_pd100))
      print row >> csv_path
      close(csv_path)
      printf("CSV: %s\n", csv_path)
    }
  '

  echo
  echo "--- Finished pipeline for MOA: ${moa} ---"
  echo
done

echo "######################################################################"
echo "###   All MOAs processing is complete!                             ###"
echo "######################################################################"