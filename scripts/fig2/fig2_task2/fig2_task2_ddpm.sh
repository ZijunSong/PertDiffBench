#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo ERROR && exit 1' ERR
export LC_ALL=C LC_NUMERIC=C

# -------------------- Config --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TARGET_CELL_TYPES=( "B" "NK" )
NUM_GENES="${NUM_GENES:-6998}"
N_SAMPLES="${N_SAMPLES:-54}"
NUM_RUNS=3
CONFIG_FILE="${CONFIG_FILE:-configs/baselines/scrna_ddpm_scrna.yaml}"
METHOD_NAME="${METHOD_NAME:-scrna_ddpm_scrna}"

# 彻底关闭 W&B
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# ---------------- Project Root ------------------
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/.."
cd "$HOMEDIR"
echo "PWD: $(pwd)"

# ---------------- Paths -------------------------
TRAIN_H5="data/fig1/raw_task1/task1_train_CD4T_exp.h5ad"
CKPT_ROOT="checkpoints/fig2/task2/pretrain_CD4T/${METHOD_NAME}"
OUT_ROOT="samples/fig2/task2_unseen_celltype/pretrain_CD4T/${METHOD_NAME}"
mkdir -p "${CKPT_ROOT}" "${OUT_ROOT}"

# 全局 CSV（聚合所有 target 的 3 次评测）
GLOBAL_CSV="${OUT_ROOT}/metrics_all.csv"
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

# ================== 3x（训练+测评） ==================
for (( run=1; run<=NUM_RUNS; run++ )); do
  echo
  echo "======================"
  echo " Run ${run}/${NUM_RUNS}  (train on CD4T, then eval targets)"
  echo "======================"

  RUN_CKPT_DIR="${CKPT_ROOT}/run${run}"
  RUN_OUT_DIR="${OUT_ROOT}/run${run}"
  mkdir -p "${RUN_CKPT_DIR}" "${RUN_OUT_DIR}"

  CKPT_PATH="${RUN_CKPT_DIR}/scrna_ddpm_epoch1000.pt"   # 按你给的 ckpt 命名

  # ---- Step 1: 训练（CD4T 预训练）----
  echo "######################################################################"
  echo "###   Step 1 (Run ${run}): Training on pretrain_CD4T"
  echo "######################################################################"
  python scripts/baseline/train_scrna_ddpm_scrna.py \
    --config "${CONFIG_FILE}" \
    --data-path "${TRAIN_H5}" \
    --save-weight-dir "${RUN_CKPT_DIR}" \
    --gene-nums "${NUM_GENES}"

  # ---- Step 2: 对所有目标 cell 做评测（使用本 run 的 ckpt）----
  for cell_type in "${TARGET_CELL_TYPES[@]}"; do
    VALID_H5="data/fig1/raw_task1/task1_valid_${cell_type}_exp.h5ad"
    CELL_OUT_DIR="${OUT_ROOT}/${cell_type}/run${run}"
    mkdir -p "${CELL_OUT_DIR}"

    echo -e "\n######################################################################"
    echo "###   Step 2 (Run ${run}): Evaluating on target: ${cell_type}"
    echo "######################################################################"

    run_output="$(
      python scripts/baseline/eval_scrna_ddpm_scrna.py \
        --config "${CONFIG_FILE}" \
        --train-data-path "${TRAIN_H5}" \
        --data-path "${VALID_H5}" \
        --ckpt "${CKPT_PATH}" \
        --out_h5ad "${CELL_OUT_DIR}/synthetic_ifn.h5ad" \
        --gene-nums "${NUM_GENES}" \
        --umap_plot "${CELL_OUT_DIR}/umap_comparison.svg" \
        --n_samples "${N_SAMPLES}" 2>&1
    )" || true

    echo "${run_output}"

    # 仅抓取指标行，缓存在该 cell 的聚合缓冲文件
    CELL_BUF="${OUT_ROOT}/${cell_type}/_agg_buffer.txt"
    mkdir -p "$(dirname "${CELL_BUF}")"
    {
      printf "%s\n" "${run_output}" | grep -E \
        "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
        || true
      printf "\n"
    } >> "${CELL_BUF}"
  done
done

# ================== 聚合到全局 CSV ==================
for cell_type in "${TARGET_CELL_TYPES[@]}"; do
  CELL_BUF="${OUT_ROOT}/${cell_type}/_agg_buffer.txt"
  [[ -f "${CELL_BUF}" ]] || { echo "[WARN] No outputs for ${cell_type}, skip."; continue; }

  awk -v ds="${cell_type}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}(${NUM_GENES})" -v csv_path="${GLOBAL_CSV}" '
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
  ' "${CELL_BUF}"
done

echo "######################################################################"
echo "###   Done! CSV => ${GLOBAL_CSV}"
echo "######################################################################"
