#!/usr/bin/env bash
# 留一法 (leave-one-out)：每次留一个物种做测试，其余物种合并做训练。scDiff 需同一 DATA_ROOT 下同时有 merged 与 test h5ad。
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME="${NAME:-v7.5}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
NUM_RUNS="${NUM_RUNS:-3}"
N_SAMPLES="${N_SAMPLES:-1000}"
METHOD_NAME="${METHOD_NAME:-scDiff}"

DATA_ROOT="data/fig2/task3_cross_species"
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
HOMEDIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

mkdir -p "$DATA_ROOT"

for test_species in "mouse" "pig" "rabbit" "rat"; do
  train_species_list=()
  for s in mouse pig rabbit rat; do
    [[ "$s" != "$test_species" ]] && train_species_list+=( "$s" )
  done
  train_species_comma=$(IFS=,; echo "${train_species_list[*]}")

  MERGED_TRAIN="${DATA_ROOT}/merged_train_${test_species}.h5ad"
  if [[ ! -f "${MERGED_TRAIN}" ]]; then
    echo "### Building merged train (leave-out ${test_species}): ${train_species_comma} -> ${MERGED_TRAIN}"
    python scripts/fig2/fig2_task3_extend/merge_species_control_ifn.py \
      --data-root "${DATA_ROOT}" \
      --train-species "${train_species_comma}" \
      --out "${MERGED_TRAIN}"
  fi

  dataset_name="${test_species}_control_ifn"
  train_fname="merged_train_${test_species}.h5ad"
  test_fname="${test_species}_control_ifn.h5ad"

  echo "######################################################################"
  echo "###   Starting pipeline for leave-one-out test species: ${test_species}"
  echo "######################################################################"

  data_settings=()
  data_settings+=("data.params.train.params.dataset=${dataset_name}")
  data_settings+=("data.params.train.params.fname=${train_fname}")
  data_settings+=("data.params.test.params.dataset=${dataset_name}")
  data_settings+=("data.params.test.params.fname=${test_fname}")

  OUT_ROOT="samples/fig2/task3_extend/scdiff/${test_species}"
  LOG_ROOT="logs/fig2/task3_extend/scdiff/${test_species}"
  METRICS_CSV="${OUT_ROOT}/metrics_leave1out_${test_species}.csv"
  mkdir -p "$OUT_ROOT" "$LOG_ROOT"

  ALL_OUTPUTS=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo -e "\n--- Run ${i}/${NUM_RUNS} for leave1out_${test_species} ---"
    if ! EVAL_OUTPUT=$(python src/scDiff/main.py \
          --custom_data_path "${DATA_ROOT}" \
          --base configs/scdiff/eval_perturbation.yaml \
          --name "${NAME}" \
          --logdir "${LOGDIR}" \
          --postfix "perturbation_${NAME}" \
          ${OFFLINE_SETTINGS} \
          "${data_settings[@]}" \
          2>&1); then
      echo "[ERROR] pipeline failed for species=${test_species} run=${i}" >&2
      echo "-------- Python Traceback --------"
      echo "$EVAL_OUTPUT" >&2
      echo "---------------------------------"
      exit 1
    fi

    printf "%s\n" "$EVAL_OUTPUT"
    run_tmp="$(mktemp)"
    pattern_re='Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)'
    grep -E "$pattern_re" <<< "$EVAL_OUTPUT" > "$run_tmp" || true
    ALL_OUTPUTS+=$(cat "$run_tmp")
    ALL_OUTPUTS+=$'\n'
    rm -f "$run_tmp"
  done

  awk -v ds="Leave1out_test_${test_species}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${METRICS_CSV}" '
BEGIN{ c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=c_p_all=c_pd_all=c_pd20=c_pd50=c_pd100=0 }
function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }
$0 ~ /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF); next }
$0 ~ /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF); next }
$0 ~ /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF); next }
$0 ~ /^E-Distance:/                              { edist[c_edist++] = to_num($NF); next }
$0 ~ /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++]  = to_num($NF); next }
$0 ~ /R-squared \(R2\):/                         { r2[c_r2++]    = to_num($NF); next }
$0 ~ /Pearson \(all genes\):/                    { p_all[c_p_all++] = to_num($NF); next }
$0 ~ /Pearson Delta \(all genes\):/              { pd_all[c_pd_all++] = to_num($NF); next }
$0 ~ /Pearson Delta \(top 20 DE genes\):/        { pd20[c_pd20++] = to_num($NF); next }
$0 ~ /Pearson Delta \(top 50 DE genes\):/        { pd50[c_pd50++] = to_num($NF); next }
$0 ~ /Pearson Delta \(top 100 DE genes\):/       { pd100[c_pd100++] = to_num($NF); next }
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
  metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES"; metric_names[4]="E-Distance"; metric_names[5]="MMD";
  metric_names[6]="R2"; metric_names[7]="Pearson (all genes)"; metric_names[8]="Pearson Delta (all genes)";
  metric_names[9]="Pearson Delta (top 20 DE genes)"; metric_names[10]="Pearson Delta (top 50 DE genes)"; metric_names[11]="Pearson Delta (top 100 DE genes)";
  header="Dataset,Method"; for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)"; for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];
  row=ds "," method; for(i=1;i<=11;i++) row=row "," mean_std(i); for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);
  print header > csv_path; print row >> csv_path; close(csv_path); printf("CSV written: %s\n", csv_path);
}
' <<< "$ALL_OUTPUTS"

  echo -e "\n--- Finished pipeline: leave1out_${test_species} ---\n"
done

echo "######################################################################"
echo "###   All leave-one-out folds complete!                             ###"
echo "######################################################################"
