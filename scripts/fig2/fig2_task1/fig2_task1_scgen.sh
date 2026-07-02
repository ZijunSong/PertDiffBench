#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/max_n_samples.sh"
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# ------------------------- Config -------------------------
SEEDS=( '123' '345' '567' )
NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-scGen}"

# ( seed)
GLOBAL_SUMMARY="samples/fig2/task1/scgen/summary_all_seeds.csv"
mkdir -p "$(dirname "$GLOBAL_SUMMARY")"

# ------------------------- Main ---------------------------
for seed in "${SEEDS[@]}"; do
  dataset_base="seed${seed}_control"
  train_ds="seed${seed}_control_train"
  test_ds="seed${seed}_control_test"

  N_SAMPLES="$(max_n_samples_paired "data/fig2/task1_unseen_pert/${test_ds}.h5ad")"

  LOG_ROOT="logs/fig2/task1/scgen/seed${seed}"
  OUT_ROOT="samples/fig2/task1/scgen/seed${seed}"
  CKPT_ROOT="checkpoints/fig2/task1/scgen/seed${seed}/${train_ds}"

  # each seed separate CSV ( )
  CSV_PATH="${OUT_ROOT}/metrics_${METHOD_NAME}_${test_ds}.csv"

  mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$CKPT_ROOT"

  echo "######################################################################"
  echo "###   Dataset: ${dataset_base}   (runs=${NUM_RUNS})"
  echo "######################################################################"

  all_outputs=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    export RUN_SEED=$(($i-1))
    run_tag="run${i}"
    run_dir="${OUT_ROOT}/${run_tag}"
    run_ckpt="${CKPT_ROOT}/${run_tag}"
    mkdir -p "$run_dir" "$run_ckpt"

    log_file="${LOG_ROOT}/${test_ds}_${run_tag}.log"
    echo
    echo "======================"
    echo " ${run_tag}/${NUM_RUNS}: TRAIN+EVAL (${train_ds} -> ${test_ds})"
    echo "======================"

    # Python stdout/stderr, one-time awk
    output=$(
      python scripts/scGen_eval.py \
        --train_data_path "data/fig2/task1_unseen_pert/${train_ds}.h5ad" \
        --test_data_path  "data/fig2/task1_unseen_pert/${test_ds}.h5ad" \
        --model_save_path "${run_ckpt}" \
        --out_h5ad       "${run_dir}/${test_ds}_pred_${i}.h5ad" \
        --umap_plot      "${run_dir}/${test_ds}_umap_${i}.png" \
        --n_samples      ""${N_SAMPLES}"" \
        --celltype_to_predict "mammary epithelial cells" \
        2>&1
    ) || true

    # parse
    all_outputs+="$output"$'\n'
  done

  # ===== parseand seed CSV ( " " ) =====
  echo -e "$all_outputs" | awk -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
    function to_num(x,   y){ y=x; gsub(/[^0-9eE+\-\.]/,"",y); return (y==""?0:y)+0 }

    # 11 items ( before andempty /when )
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
    function ms(a,n,   mu,sd){ mu=mean(a,n); sd=std(a,n); return sprintf("%.4f±%.4f",mu,sd) }
    function join_runs(a,n, nr,  r,res){ res=""; for(r=0;r<nr;r++) res = res sprintf(",%.4f", (r in a ? a[r] : 0)); return res }

    END{
      # colsname (and , can )
      metric_names[1]="PDS"
      metric_names[2]="MAE"
      metric_names[3]="DES"
      metric_names[4]="E-Distance"
      metric_names[5]="MMD"
      metric_names[6]="R2"
      metric_names[7]="Pearson (all genes)"
      metric_names[8]="Pearson Delta (all genes)"
      metric_names[9]="Pearson Delta (top 20 DE genes)"
      metric_names[10]="Pearson Delta (top 50 DE genes)"
      metric_names[11]="Pearson Delta (top 100 DE genes)"

      header="Method"
      for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)"
      for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i]

      row=method
      row=row "," ms(pds,  c_pds)
      row=row "," ms(mae,  c_mae)
      row=row "," ms(des,  c_des)
      row=row "," ms(edist,c_edist)
      row=row "," ms(mmd,  c_mmd)
      row=row "," ms(r2,   c_r2)
      row=row "," ms(p_all,c_pa)
      row=row "," ms(pd_all,c_pda)
      row=row "," ms(pd20, c_pd20)
      row=row "," ms(pd50, c_pd50)
      row=row "," ms(pd100,c_pd100)

      # run originalvalue ( num_runs 0)
      row=row join_runs(pds,   c_pds,   num_runs)
      row=row join_runs(mae,   c_mae,   num_runs)
      row=row join_runs(des,   c_des,   num_runs)
      row=row join_runs(edist, c_edist, num_runs)
      row=row join_runs(mmd,   c_mmd,   num_runs)
      row=row join_runs(r2,    c_r2,    num_runs)
      row=row join_runs(p_all, c_pa,    num_runs)
      row=row join_runs(pd_all,c_pda,   num_runs)
      row=row join_runs(pd20,  c_pd20,  num_runs)
      row=row join_runs(pd50,  c_pd50,  num_runs)
      row=row join_runs(pd100, c_pd100, num_runs)

      print header > csv_path
      print row    >> csv_path
      close(csv_path)
      printf("CSV written: %s\n", csv_path) > "/dev/stderr"
    }
  '


  echo
  echo "--- Finished all runs for seed${seed} (${test_ds}) ---"
  echo
done

echo "######################################################################"
echo "###   All seeds completed!"
echo "###   Global summary: ${GLOBAL_SUMMARY}"
echo "######################################################################"
