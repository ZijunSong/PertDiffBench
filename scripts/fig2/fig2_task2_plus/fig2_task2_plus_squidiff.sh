#!/usr/bin/env bash
# Squidiff: train on scGen-style combined h5ad; evaluate for each LOO fold x control fraction.
# Default: p0 only. For p0.25 / p0.5 on separate GPUs, use:
#   fig2_task2_plus_squidiff_p0.25.sh (default GPU 6), fig2_task2_plus_squidiff_p0.5.sh (default GPU 7).
# Or: export FIG2_TASK2_PLUS_SQUIDIFF_SLUGS="p0.25 p0.5" (space-separated; matches loo_<CT>/<slug>/).
# Optional: FIG2_TASK2_PLUS_REVERSE_HOLDOUT=1 → iterate HOLDOUT_TYPES as NK … B (reverse of default).
# Optional: FIG2_TASK2_PLUS_HOLDOUT_TYPES="CD14+Mono Dendritic" → only these holdouts (space-separated;
# names must match default list: B, CD4T, CD8T, CD14+Mono, Dendritic, FCGR3A+Mono, NK).
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C
# Default single-GPU (override with CUDA_VISIBLE_DEVICES). Split scripts default to GPU 6 / 7.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

GENE_SIZE="${GENE_SIZE:-6998}"
NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-Squidiff}"
# Eval pairs = min(#Control, #pert) on test h5ad; global minimum in task2+ is 259 (NK @ p0.5). Keep <=259.
N_SAMPLES="${N_SAMPLES:-256}"

HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"

DATA_BASE="${DATA_BASE:-data/fig2/task2_unseen_celltype_plus}"

if [[ -n "${FIG2_TASK2_PLUS_HOLDOUT_TYPES:-}" ]]; then
  read -r -a HOLDOUT_TYPES <<< "${FIG2_TASK2_PLUS_HOLDOUT_TYPES}"
  echo "[fig2_task2_plus_squidiff] HOLDOUT_TYPES (override): ${HOLDOUT_TYPES[*]}"
else
  HOLDOUT_TYPES=( "B" "CD4T" "CD8T" "CD14+Mono" "Dendritic" "FCGR3A+Mono" "NK" )
fi
if [[ "${FIG2_TASK2_PLUS_REVERSE_HOLDOUT:-}" == "1" ]]; then
  _rev_ht=()
  for ((_j=${#HOLDOUT_TYPES[@]}-1; _j>=0; _j--)); do
    _rev_ht+=("${HOLDOUT_TYPES[_j]}")
  done
  HOLDOUT_TYPES=("${_rev_ht[@]}")
  echo "[fig2_task2_plus_squidiff] HOLDOUT_TYPES (reversed): ${HOLDOUT_TYPES[*]}"
fi
if [[ -n "${FIG2_TASK2_PLUS_SQUIDIFF_SLUGS:-}" ]]; then
  read -r -a CTRL_SLUGS <<< "${FIG2_TASK2_PLUS_SQUIDIFF_SLUGS}"
else
  CTRL_SLUGS=( "p0" )
fi

sample_base="samples/fig2/fig2_task2_plus/squidiff"
ckpt_base="checkpoints/fig2/fig2_task2_plus/pretrain/squidiff"
GLOBAL_CSV="${sample_base}/metrics_all.csv"
mkdir -p "${sample_base}" "${ckpt_base}"

METRICS_LOCK="${sample_base}/.metrics_all_csv.lock"
METRICS_ROWS_LOCK="${sample_base}/.metrics_rows.lock"
(
  flock -w 120 200 || true
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
) 200>"${METRICS_LOCK}"

cd src/Squidiff

for ht in "${HOLDOUT_TYPES[@]}"; do
  for slug in "${CTRL_SLUGS[@]}"; do
    REL_COMBINED="${DATA_BASE}/loo_${ht}/${slug}/scgen_combined_train_plus_test_control.h5ad"
    REL_TEST="${DATA_BASE}/loo_${ht}/${slug}/task2_test_exp.h5ad"
    COMBINED_H5="../../${REL_COMBINED}"
    TEST_H5="../../${REL_TEST}"
    ds_tag="${ht}_${slug}"

    if [[ ! -f "${COMBINED_H5}" || ! -f "${TEST_H5}" ]]; then
      echo "[WARN] Skip ${ds_tag}: missing ${REL_COMBINED} or ${REL_TEST}"
      continue
    fi

    CELL_BUF="../../${sample_base}/${ht}/${slug}/_agg_buffer.txt"
    mkdir -p "$(dirname "${CELL_BUF}")"
    : > "${CELL_BUF}"

    echo "######################################################################"
    echo "###   Squidiff | ${ds_tag}"
    echo "######################################################################"

    for (( i=1; i<=NUM_RUNS; i++ )); do
      run_ckpt_dir="../../${ckpt_base}/${ht}/${slug}/run${i}"
      run_sample_dir="../../${sample_base}/${ht}/${slug}/run${i}"
      mkdir -p "${run_ckpt_dir}" "${run_sample_dir}"

      echo "--- Run ${i}/${NUM_RUNS}: Training ${ds_tag} ---"
      python train_squidiff.py \
        --logger_path "../../logs/squidiff/fig2_task2_plus/${ds_tag}_g${GENE_SIZE}_run${i}" \
        --data_path "${COMBINED_H5}" \
        --resume_checkpoint "${run_ckpt_dir}" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${GENE_SIZE}"

      echo "--- Run ${i}/${NUM_RUNS}: Sampling ${ds_tag} ---"
      run_output=""
      run_output=$(python sample_squidiff.py \
        --model_path "${run_ckpt_dir}/model.pt" \
        --gene_size "${GENE_SIZE}" \
        --output_dim "${GENE_SIZE}" \
        --out_h5ad "${run_sample_dir}/synthetic_ifn.h5ad" \
        --n_samples "${N_SAMPLES}" \
        --train_data_path "${COMBINED_H5}" \
        --data_path "${TEST_H5}" 2>&1) || true
      echo "${run_output}"
      {
        printf "%s\n" "${run_output}" | grep -E \
          "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
          || true
        printf "\n"
      } >> "${CELL_BUF}"
    done

    (
      flock -w 3600 199 || true
      awk -v ds="${ds_tag}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="../../${GLOBAL_CSV}" '
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
    # cwd 为 src/Squidiff；锁路径须相对 HOMEDIR，否则会出现 .metrics_rows.lock: No such file or directory 并提前退出，导致只跑完第一个 holdout。
    ) 199>"${HOMEDIR}/${METRICS_ROWS_LOCK}"

    echo "--- Finished ${ds_tag} ---"
  done
done

cd "${HOMEDIR}"
echo "######################################################################"
echo "###   Squidiff fig2_task2_plus done! CSV => ${GLOBAL_CSV}"
echo "######################################################################"
