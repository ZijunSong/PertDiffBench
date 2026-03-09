#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C LC_NUMERIC=C

# ================= Script Configuration =================
CELL_TYPES=('CD4T')
NOISE_LEVELS=('0.1' '0.25' '0.5' '1.0' '1.5')

NUM_GENES="${NUM_GENES:-6998}"
NUM_RUNS="${NUM_RUNS:-3}"

CONFIG_FILE="${CONFIG_FILE:-configs/baselines/scrna_ddpm_scrna.yaml}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/add_lognormal_bionoise_output}"

# 用于写入 CSV 的方法名（表头里的 Method 字段）
METHOD_NAME="${METHOD_NAME:-scRNA-DDPM-scRNA}"
# 用于路径的最后一级目录名（不同方法只改这一层即可）
METHOD_DIR="${METHOD_DIR:-scrna_ddpm_scrna}"

BASE_CKPT_DIR="${BASE_CKPT_DIR:-checkpoints/lognormal_bionoise}"
BASE_SAMPLES_DIR="${BASE_SAMPLES_DIR:-samples/lognormal_bionoise}"

mkdir -p logs

# ================= Main Processing Loop =================
for cell_type in "${CELL_TYPES[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###   Pipeline: Cell=${cell_type} | Noise Std=${noise_level}"
    echo "######################################################################"

    # ---- Dynamic Paths (data) ----
    train_data_path="${BASE_DATA_DIR}/task1_train_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"
    valid_data_path="${BASE_DATA_DIR}/task1_valid_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"

    # group_suffix 只跟 cell_type + noise 有关
    group_suffix="${cell_type}_noise_${noise_level}"

    # 这里的大路径只到 group_suffix，最后一层用 METHOD_DIR 区分不同方法
    base_weight_dir="${BASE_CKPT_DIR}/${group_suffix}/${METHOD_DIR}"
    base_samples_dir="${BASE_SAMPLES_DIR}/${group_suffix}/${METHOD_DIR}"

    mkdir -p "${base_weight_dir}" "${base_samples_dir}"

    echo -e "\n--- Train + Eval ${NUM_RUNS} runs (${cell_type}, noise=${noise_level}) ---"

    # 收集所有 run 的指标行，用来后面 awk 聚合
    ALL_OUTPUTS=""

    # 统一的 grep pattern，用来从 eval 输出里抽指标行
    pattern_re='Perturbation Discrimination Score \(PDS\)|Mean Absolute Error \(MAE\)|Differential Expression Score \(DES\)|^E-Distance:|Maximum Mean Discrepancy \(MMD\)|R-squared \(R2\)|Pearson \(all genes\)|Pearson Delta \(all genes\)|Pearson Delta \(top 20 DE genes\)|Pearson Delta \(top 50 DE genes\)|Pearson Delta \(top 100 DE genes\)'

    for (( run_id=1; run_id<=NUM_RUNS; run_id++ )); do
      echo -e "\n================ Run ${run_id}/${NUM_RUNS} ================"

      # 每个 run 自己的目录，仍然挂在同一个方法目录下面
      run_weight_dir="${base_weight_dir}/run_${run_id}"
      run_samples_dir="${base_samples_dir}/run_${run_id}"
      mkdir -p "${run_weight_dir}" "${run_samples_dir}"

      checkpoint_file="${run_weight_dir}/scrna_ddpm_epoch1000.pt"

      # ---- Step 1: Train (per run) ----
      echo -e "\n--- [Run ${run_id}] Train ---"
      python scripts/baseline/train_scrna_ddpm_scrna.py \
        --config "${CONFIG_FILE}" \
        --data-path "${train_data_path}" \
        --save-weight-dir "${run_weight_dir}" \
        --gene-nums "${NUM_GENES}" 2>&1 | tee "logs/train_${cell_type}_noise_${noise_level}_run${run_id}.log"

      if [[ ! -f "${checkpoint_file}" ]]; then
        echo "[ERROR] checkpoint not found after training: ${checkpoint_file}" >&2
        exit 1
      fi
      echo "--- [Run ${run_id}] Train done, ckpt=${checkpoint_file} ---"

      # ---- Step 2: Eval (per run) ----
      echo -e "\n--- [Run ${run_id}] Eval ---"
      EVAL_OUTPUT="$(
        python scripts/baseline/eval_scrna_ddpm_scrna.py \
          --config "${CONFIG_FILE}" \
          --data-path "${valid_data_path}" \
          --train-data-path "${train_data_path}" \
          --ckpt "${checkpoint_file}" \
          --out_h5ad "${run_samples_dir}/synthetic_ifn_run_${run_id}.h5ad" \
          --gene-nums "${NUM_GENES}" \
          --umap_plot "${run_samples_dir}/umap_comparison_run_${run_id}.png" \
          --n_samples "${N_SAMPLES:-6}" \
          2>&1
      )"
      status=$?

      # 完整输出到 log，方便 debug
      printf "%s\n" "${EVAL_OUTPUT}" | tee "logs/eval_${cell_type}_noise_${noise_level}_run${run_id}.log"

      if (( status != 0 )); then
        echo "[ERROR] evaluation failed at run ${run_id}. Full traceback above." >&2
        exit "${status}"
      fi

      # 只抽取包含指标的行，拼到 ALL_OUTPUTS 中
      run_tmp="$(mktemp)"
      grep -E "${pattern_re}" <<< "${EVAL_OUTPUT}" > "${run_tmp}" || true
      ALL_OUTPUTS+=$(cat "${run_tmp}")
      ALL_OUTPUTS+=$'\n'
      rm -f "${run_tmp}"

      echo -e "--- [Run ${run_id}] Done ---\n"
    done

    # ---- Step 3: Aggregate -> CSV (per cell_type, noise_level) ----
    metrics_csv="${base_samples_dir}/metrics_${group_suffix}.csv"
    echo -e "\n--- Step 3: Aggregate metrics -> CSV (${metrics_csv}) ---"

    awk -v ds="${cell_type}" -v nz="${noise_level}" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${metrics_csv}" '
BEGIN{
  c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=c_p_all=c_pd_all=c_pd20=c_pd50=c_pd100=0
}
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

  header="Dataset,Noise,Method";
  for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
  for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

  row=ds "," nz "," method;
  for(i=1;i<=11;i++) row=row "," mean_std(i);
  for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

  print header > csv_path;
  print row    >> csv_path;
  close(csv_path);
  printf("CSV written: %s\n", csv_path);
}
' <<< "${ALL_OUTPUTS}"

    echo -e "\n--- Finished: Cell=${cell_type} | Noise=${noise_level} ---\n"
  done
done

echo "######################################################################"
echo "###   All processing complete                                       ###"
echo "######################################################################"
