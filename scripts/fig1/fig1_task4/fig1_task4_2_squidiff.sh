#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# ------------------------- Config -------------------------
# prefix → gene_size
declare -A DATASETS=(
  ["ACTA2"]="5737"
  ["B2M"]="5737"
)

NUM_RUNS="${NUM_RUNS:-3}" # train+eval count
N_SAMPLES="${N_SAMPLES:-100}" # each runeval countamount
METHOD_NAME="${METHOD_NAME:-Squidiff}"

echo "Changing directory to src/Squidiff..."
cd src/Squidiff

# ------------------------- Main ---------------------------
for prefix in "${!DATASETS[@]}"; do
  gene_size="${DATASETS[$prefix]}"

  train_dataset="task4_${prefix}_control_to_ifn"
  eval_dataset="task4_${prefix}_control_to_coculture"

  LOG_ROOT="../../logs/fig1/task4_2/${eval_dataset}/squidiff_${gene_size}"
  OUT_ROOT="../../samples/fig1/task4_2/${prefix}/squidiff"
  CKPT_ROOT="../../checkpoints/fig1/task4_2/${prefix}_control_to_ifn/squidiff_${gene_size}"
  METRICS_CSV="${OUT_ROOT}/metrics_${eval_dataset}.csv"

  mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$CKPT_ROOT"

  echo "######################################################################"
  echo "### Prefix: ${prefix}"
  echo "### Train on: ${train_dataset}"
  echo "### Eval  on: ${eval_dataset}   (runs=${NUM_RUNS}, gene_size=${gene_size}, n_samples=${N_SAMPLES})"
  echo "######################################################################"

  ALL_OUTPUTS=""

  # -------- Run 1..NUM_RUNS (each run run train+eval) --------
  for (( i=1; i<=NUM_RUNS; i++ )); do
    run_tag="run${i}"
    run_dir="${OUT_ROOT}/${run_tag}"
    mkdir -p "$run_dir"

    # aseach run run preparestandalone ckpt directory, 
    run_ckpt_dir="${CKPT_ROOT}/${run_tag}"
    mkdir -p "$run_ckpt_dir"

    log_file="${LOG_ROOT}/${eval_dataset}_${run_tag}.log"
    echo "[$(date '+%F %T')] >>> ${run_tag}: TRAIN (${train_dataset})" | tee "$log_file"

    # -------------------- Step 1: Train --------------------
    # train_squidiff.py will to --resume_checkpoint specifydirectory ( model.pt)
    python train_squidiff.py \
      --logger_path "../../logs/squidiff/task4/${train_dataset}/${run_tag}" \
      --data_path   "../../data/fig1/task4/${train_dataset}.h5ad" \
      --resume_checkpoint "${run_ckpt_dir}" \
      --gene_size   "$gene_size" \
      --output_dim  "$gene_size" \
      2>&1 | tee -a "$log_file"

    # -------------------- Step 2: Sample + Eval --------------------
    echo "[$(date '+%F %T')] >>> ${run_tag}: EVAL  (${eval_dataset})" | tee -a "$log_file"

    # andeval willin stdout 11 items ; here when 
    if python sample_squidiff.py \
        --model_path "${run_ckpt_dir}/model.pt" \
        --gene_size  "$gene_size" \
        --output_dim "$gene_size" \
        --out_h5ad   "${run_dir}/${eval_dataset}_synthetic_ifn_${i}.h5ad" \
        --n_samples  "${N_SAMPLES}" \
        --umap_plot  "${run_dir}/${eval_dataset}_umap_${i}.png" \
        --train_data_path "../../data/fig1/task4/${train_dataset}.h5ad" \
        --data_path  "../../data/fig1/task4/${eval_dataset}.h5ad" \
        2>&1 | tee -a "$log_file"
    then
      :
    else
      echo "[ERROR] ${run_tag} eval failed. See log: ${log_file}" >&2
      exit 1
    fi

    # canparse , to ALL_OUTPUTS
    run_tmp="$(mktemp)"
    grep -E "Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)" \
      "$log_file" > "$run_tmp" || true
    ALL_OUTPUTS+=$(cat "$run_tmp")
    ALL_OUTPUTS+=$'\n'
    rm -f "$run_tmp"
  done

  # -------- Step 3: CSV (mean±std + each runoriginalvalue) --------
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
      print row >> csv_path; # 
      close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '

  echo -e "\n--- Finished all runs for ${prefix} (${eval_dataset}) ---\n"
done

echo "######################################################################"
echo "###   All prefixes completed!                                      ###"
echo "######################################################################"
