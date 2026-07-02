#!/usr/bin/env bash
# scGen for Fig2 task2+: leave-one-out over 7 cell types x held-out control fractions (0% / 25% / 50%).
# Run preprocess_data/fig2/task2_unseen_celltype_plus/task2_unseen_celltype_plus_loo.py first to generate .h5ad files.
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C
# Default single-GPU: device 0 (override with CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-scGen}"
# Eval pairs = min(#Control, #pert) on test h5ad; global minimum in task2+ is 259 (NK @ p0.5). Keep <=259.


HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"
echo "PWD: $(pwd)"

DATA_BASE="${DATA_BASE:-data/fig2/task2_unseen_celltype_plus}"
DATA_ROOT="${HOMEDIR}/${DATA_BASE}"

HOLDOUT_TYPES=( "B" "CD4T" "CD8T" "CD14+Mono" "Dendritic" "FCGR3A+Mono" "NK" )
CTRL_SLUGS=( "p0" "p0.25" "p0.5" )

OUT_ROOT="samples/fig2/fig2_task2_plus/scgen"
CKPT_ROOT="checkpoints/fig2/fig2_task2_plus/scgen"
GLOBAL_CSV="${OUT_ROOT}/metrics_all.csv"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"

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
    DATA_DIR="${DATA_ROOT}/loo_${ht}/${slug}"
    TRAIN_H5="${DATA_DIR}/task2_train_exp.h5ad"
    TEST_H5="${DATA_DIR}/task2_test_exp.h5ad"
    ds_tag="${ht}_${slug}"

    if [[ ! -f "${TRAIN_H5}" || ! -f "${TEST_H5}" ]]; then
      echo "[WARN] Skip ${ds_tag}: missing ${TRAIN_H5} or ${TEST_H5}"
      continue
    fi

    cell_out="${OUT_ROOT}/${ht}/${slug}"
    cell_ckpt="${CKPT_ROOT}/${ht}/${slug}"
    mkdir -p "$cell_out" "$cell_ckpt"

    echo "######################################################################"
    echo "###   ${ds_tag}  |  train+eval scGen  |  runs=${NUM_RUNS}"
    echo "######################################################################"

    all_outputs=""
    for (( i=1; i<=NUM_RUNS; i++ )); do
      export RUN_SEED=$(($i-1))
      run_tag="run${i}"
      run_dir="${cell_out}/${run_tag}"
      run_ckpt="${cell_ckpt}/${run_tag}"
      mkdir -p "$run_dir" "$run_ckpt"

      echo
      echo "======================"
      echo " ${run_tag}/${NUM_RUNS}: TRAIN+EVAL  ${ds_tag}"
      echo "======================"

      output=$(
        python scripts/scGen_eval.py \
          --train_data_path "${TRAIN_H5}" \
          --test_data_path  "${TEST_H5}" \
          --model_save_path "${run_ckpt}" \
          --out_h5ad       "${run_dir}/task2_test_pred_${i}.h5ad" \
          --umap_plot      "${run_dir}/task2_umap_${i}.png" \
          --n_samples      "${N_SAMPLES}" \
          --celltype_to_predict "${ht}" \
          2>&1
      ) || true

      all_outputs+="$output"$'\n'
      printf "%s\n" "$output"
    done

    echo -e "$all_outputs" | awk -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v ds="${ds_tag}" -v csv_path="${GLOBAL_CSV}" '
      function to_num(x,   y){ y=x; gsub(/[^0-9eE+\-\.]/,"",y); return (y==""?0:y)+0 }

      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++]     = to_num($NF) }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++]     = to_num($NF) }
      /Differential Expression Score \(DES\):/    { des[c_des++]     = to_num($NF) }
      /[[:space:]]E-?Distance:/                   { edist[c_edist++] = to_num($NF) }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++]     = to_num($NF) }
      /R-?squared \(R2\):/                        { r2[c_r2++]       = to_num($NF) }
      /Pearson \(all genes\):/                    { p_all[c_pa++]    = to_num($NF) }
      /Pearson Delta \(all genes\):/              { pd_all[c_pda++]  = to_num($NF) }
      /Pearson Delta \(top 20 DE genes\):/        { pd20[c_pd20++]   = to_num($NF) }
      /Pearson Delta \(top 50 DE genes\):/        { pd50[c_pd50++]   = to_num($NF) }
      /Pearson Delta \(top 100 DE genes\):/       { pd100[c_pd100++] = to_num($NF) }

      function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n : 0 }
      function std(a,n,  mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }
      function ms(a,n,   mu,sd){ mu=mean(a,n); sd=std(a,n); return sprintf("%.6f±%.6f",mu,sd) }
      function val(idx,r,  v){
        if(idx==1) v=pds[r]; else if(idx==2) v=mae[r]; else if(idx==3) v=des[r]; else if(idx==4) v=edist[r];
        else if(idx==5) v=mmd[r]; else if(idx==6) v=r2[r]; else if(idx==7) v=p_all[r]; else if(idx==8) v=pd_all[r];
        else if(idx==9) v=pd20[r]; else if(idx==10) v=pd50[r]; else if(idx==11) v=pd100[r];
        return (v==""?0:v)+0;
      }

      END{
        row=ds "," method;
        row=row "," ms(pds,c_pds) "," ms(mae,c_mae) "," ms(des,c_des) "," ms(edist,c_edist) "," ms(mmd,c_mmd) "," ms(r2,c_r2);
        row=row "," ms(p_all,c_pa) "," ms(pd_all,c_pda) "," ms(pd20,c_pd20) "," ms(pd50,c_pd50) "," ms(pd100,c_pd100);
        for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r));
        print row >> csv_path;
        close(csv_path);
        printf("CSV appended: %s\n", csv_path) > "/dev/stderr";
      }
    '

    echo "--- Finished ${ds_tag} ---"
  done
done

echo "######################################################################"
echo "###   scGen fig2_task2_plus done! CSV => ${GLOBAL_CSV}"
echo "######################################################################"
