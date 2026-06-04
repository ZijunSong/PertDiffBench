#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# ------------------------- Config -------------------------
PREFIXES=( "ACTA2" "B2M" )

declare -A SAMPLE_SIZES=(
  ["ACTA2"]="408"
  ["B2M"]="367"
)

NUM_GENES="${NUM_GENES:-5737}"
NUM_RUNS="${NUM_RUNS:-3}"
CONFIG_FILE="${CONFIG_FILE:-configs/baselines/mlp_ddpm_mlp.yaml}"
METHOD_NAME="${METHOD_NAME:-MLP-DDPM-MLP}"

for prefix in "${PREFIXES[@]}"; do
  n_samples=${SAMPLE_SIZES[$prefix]}

  # train: control->ifn; eval: control->coculture ( for)
  train_dataset="task4_${prefix}_control_to_ifn"
  eval_dataset="task4_${prefix}_control_to_coculture"

  LOG_ROOT="logs/fig1/task4_2/${eval_dataset}/mlp_ddpm_mlp"
  CSV_ROOT="samples/fig1/task4_2/${eval_dataset}/mlp_ddpm_mlp"
  CKPT_DIR="checkpoints/fig1/task4_2/${prefix}_control_to_ifn/mlp_ddpm_mlp"
  METRICS_CSV="${CSV_ROOT}/metrics_${eval_dataset}.csv" # ✅ output ( )

  mkdir -p "$LOG_ROOT" "$CSV_ROOT" "$CKPT_DIR"

  echo "######################################################################"
  echo "###   Train on: ${train_dataset}"
  echo "###   Eval  on: ${eval_dataset}  (runs=${NUM_RUNS}, n_samples=${n_samples}, genes=${NUM_GENES})"
  echo "######################################################################"

  # -------------------- Step 1: Train (once) --------------------
  train_log="${LOG_ROOT}/${train_dataset}_train.log"
  {
    echo "[$(date '+%F %T')] >>> Training (${train_dataset})"
    python scripts/baseline/train_mlp_ddpm_mlp.py \
      --config "$CONFIG_FILE" \
      --data-path "data/fig1/task4/${train_dataset}.h5ad" \
      --save-weight-dir "$CKPT_DIR" \
      --gene-nums "$NUM_GENES"
    echo "[$(date '+%F %T')] >>> Training finished (${train_dataset})"
  } 2>&1 | tee "$train_log"

  # -------------------- Step 2: Eval (capture ONLY eval stdout) --------------------
  ALL_OUTPUTS=""
  for (( i=1; i<=NUM_RUNS; i++ )); do
    run_tag="run${i}"
    run_dir="${CSV_ROOT}/${run_tag}"
    mkdir -p "$run_dir"

    # Ensure ckpt exist
    CKPT_FILE="${CKPT_DIR}/model_epoch_1000.pth"
    if [[ ! -f "$CKPT_FILE" ]]; then
      echo "[ERROR] Checkpoint not found: ${CKPT_FILE}" >&2
      exit 1
    fi

    eval_log="${LOG_ROOT}/${eval_dataset}_${run_tag}.log"
    echo "[$(date '+%F %T')] >>> Evaluation ${run_tag} (${eval_dataset})" | tee -a "$eval_log"

    # usingtempfile run output, after awk 
    run_tmp="$(mktemp)"
    # key: mustusing ; , andexplicit 
    if python scripts/baseline/eval_mlp_ddpm_mlp.py \
        --config "$CONFIG_FILE" \
        --data-path "data/fig1/task4/${eval_dataset}.h5ad" \
        --train-data-path "data/fig1/task4/${train_dataset}.h5ad" \
        --ckpt "$CKPT_FILE" \
        --out_h5ad "${run_dir}/synthetic_ifn_${i}.h5ad" \
        --gene-nums "$NUM_GENES" \
        --umap_plot "${run_dir}/umap_comparison_${i}.png" \
        --n_samples "$n_samples" \
        2>&1 | tee -a "$eval_log" | tee "$run_tmp" > /dev/null
    then
      # onlyin when output
      ALL_OUTPUTS+=$(cat "$run_tmp")
      ALL_OUTPUTS+=$'\n'
    else
      echo "[ERROR] Evaluation ${run_tag} failed. See log: ${eval_log}" >&2
      # after run, can as 'continue'
      exit 1
    fi
    rm -f "$run_tmp"
  done

  # -------------------- Step 3: Aggregate to ONE CSV --------------------
  # evaloutput 11 ( , run ): 
  # PDS / MAE / DES / E-Distance / MMD / R2 / Pearson(all) /
  # Pearson Delta(all) / Pearson Delta(top 20 DE) / (top 50 DE) / (top 100 DE)
  echo -e "${ALL_OUTPUTS}" | awk -v ds="${eval_dataset}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${METRICS_CSV}" '
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
      metric_names[1]="PDS";
      metric_names[2]="MAE";
      metric_names[3]="DES";
      metric_names[4]="E-Distance";
      metric_names[5]="MMD";
      metric_names[6]="R2";
      metric_names[7]="Pearson (all genes)";
      metric_names[8]="Pearson Delta (all genes)";
      metric_names[9]="Pearson Delta (top 20 DE genes)";
      metric_names[10]="Pearson Delta (top 50 DE genes)";
      metric_names[11]="Pearson Delta (top 100 DE genes)";

      header="Dataset,Method";
      for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
      for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

      row=ds "," method;
      for(i=1;i<=11;i++) row=row "," mean_std(i);
      for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

      print header > csv_path; # header
      print row >> csv_path; # data
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished pipeline for ${prefix} (${eval_dataset}) ---\n"
done

echo "######################################################################"
echo "###   All prefix processing is complete!                           ###"
echo "######################################################################"
