#!/usr/bin/env bash
# scDiffusion: for each leave-one-out x control fraction, train VAE / diffusion / classifier once, then NUM_RUNS sampling runs.
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C
# Default single-GPU: device 5 (override with CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

NUM_GENES="${NUM_GENES:-6998}"
# Sampling/eval uses test controls; min usable pairs in task2+ test sets is 259 (NK @ p0.5). Keep <=259.
NUM_SAMPLES="${NUM_SAMPLES:-256}"
NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-scDiffusion}"

HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"

# Pretrained VAE encoder/decoder directory (encoder.ckpt, decoder.ckpt, gene_order.tsv).
# Override if your checkpoints live elsewhere (default matches a common external layout).
ANNOTATION_MODEL_DIR="${ANNOTATION_MODEL_DIR:-/data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1}"

DATA_BASE="${DATA_BASE:-data/fig2/task2_unseen_celltype_plus}"

HOLDOUT_TYPES=( "B" "CD4T" "CD8T" "CD14+Mono" "Dendritic" "FCGR3A+Mono" "NK" )
CTRL_SLUGS=( "p0" "p0.25" "p0.5" )

OUT_PREFIX="samples/fig2/fig2_task2_plus/scDiffusion"
GLOBAL_CSV="${OUT_PREFIX}/metrics_all.csv"
mkdir -p "${OUT_PREFIX}"

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

for ht in "${HOLDOUT_TYPES[@]}"; do
  for slug in "${CTRL_SLUGS[@]}"; do
    REL_COMBINED="${DATA_BASE}/loo_${ht}/${slug}/scgen_combined_train_plus_test_control.h5ad"
    REL_TEST="${DATA_BASE}/loo_${ht}/${slug}/task2_test_exp.h5ad"
    COMBINED_H5="${HOMEDIR}/${REL_COMBINED}"
    TEST_H5="${HOMEDIR}/${REL_TEST}"
    ds_tag="${ht}_${slug}"

    if [[ ! -f "${COMBINED_H5}" || ! -f "${TEST_H5}" ]]; then
      echo "[WARN] Skip ${ds_tag}: missing h5ad (${REL_COMBINED})"
      continue
    fi

    CKPT_PREFIX="checkpoints/fig2/fig2_task2_plus/scdiffusion/${ht}/${slug}"
    mkdir -p "${CKPT_PREFIX}"

    VAE_CKPT="${CKPT_PREFIX}/vae_checkpoint/model_seed=0_step=9999.pt"
    DIFF_CKPT="${CKPT_PREFIX}/diffusion_checkpoint/my_diffusion/model010000.pt"
    CLS_DIR="${CKPT_PREFIX}/classifier_checkpoint"
    CLS_CKPT="${CLS_DIR}/model009999.pt"

    echo "######################################################################"
    echo "###   scDiffusion train | ${ds_tag}"
    echo "######################################################################"

    echo "### Step 1: VAE"
    pushd src/scDiffusion/VAE >/dev/null
    python VAE_train.py \
      --data_dir "../../../${REL_COMBINED}" \
      --num_genes "${NUM_GENES}" \
      --state_dict "${ANNOTATION_MODEL_DIR}" \
      --save_dir "../../../${CKPT_PREFIX}/vae_checkpoint"
    popd >/dev/null

    echo "### Step 2: Diffusion"
    pushd src/scDiffusion >/dev/null
    python cell_train.py --data_dir "../../${REL_COMBINED}" --vae_path "../../${VAE_CKPT}" --save_dir "../../${CKPT_PREFIX}/diffusion_checkpoint"
    popd >/dev/null

    echo "### Step 3: Classifier"
    pushd src/scDiffusion >/dev/null
    python classifier_train.py --data_dir "../../${REL_COMBINED}" --vae_path "../../${VAE_CKPT}" --model_path "../../${CLS_DIR}"
    popd >/dev/null

    OUT_DIR="${OUT_PREFIX}/${ht}/${slug}"
    mkdir -p "${OUT_DIR}"
    ALL_OUTPUTS=""

    echo "### Step 4: Sampling (${NUM_RUNS} runs) | ${ds_tag}"
    for (( i=1; i<=NUM_RUNS; i++ )); do
      pushd src/scDiffusion >/dev/null
      run_out=""
      run_out=$(python classifier_sample.py \
        --num_samples "${NUM_SAMPLES}" \
        --train-data-path "../../${REL_COMBINED}" \
        --model_path "../../${DIFF_CKPT}" \
        --classifier_path "../../${CLS_CKPT}" \
        --ae_dir "../../${VAE_CKPT}" \
        --num_gene "${NUM_GENES}" \
        --sample_dir "../../${OUT_DIR}" \
        --out_h5ad "../../${OUT_DIR}/synthetic_ifn_${i}.h5ad" \
        --init_cell_path "../../${REL_TEST}" 2>&1) || true
      popd >/dev/null
      echo "${run_out}"
      ALL_OUTPUTS+="$(printf "%s\n" "${run_out}" | grep -E \
        "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" || true)"
      ALL_OUTPUTS+=$'\n'
    done

    printf "%s\n" "${ALL_OUTPUTS}" | awk -v ds="${ds_tag}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}(${NUM_GENES})" -v csv_path="${GLOBAL_CSV}" '
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
    '
    echo "--- Finished ${ds_tag} ---"
  done
done

echo "######################################################################"
echo "###   scDiffusion fig2_task2_plus done! CSV => ${GLOBAL_CSV}"
echo "######################################################################"
