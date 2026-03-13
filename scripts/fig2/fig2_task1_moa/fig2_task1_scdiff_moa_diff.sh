#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR
export LC_ALL=C LC_NUMERIC=C

# -------------------- Configuration --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Reduce memory during test; increase BATCH_SIZE/T_SAMPLE if OOM
BATCH_SIZE="${BATCH_SIZE:-3072}"
NUM_WORKERS="${NUM_WORKERS:-0}"
T_SAMPLE="${T_SAMPLE:-1000}"
LOGDIR="${LOGDIR:-logs}"
NAME="${NAME:-v7.5}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OFFLINE_SETTINGS="--wandb f"
NUM_RUNS="${NUM_RUNS:-3}"
METHOD_NAME="${METHOD_NAME:-scDiff}"

# -------------------- Project Root ---------------------
HOMEDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")/..
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# -------------------- Paths ----------------------------
# Preprocessed data root (control_plus_ifn per split). Checkpoints and samples use fixed base paths.
DATA_BASE="${DATA_BASE:-/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA}"
SAMPLES_BASE="${SAMPLES_BASE:-/data/ppnm/data/PertDiffBench/samples}"
CKPT_BASE="${CKPT_BASE:-/data/ppnm/checkpoints/PertDiffBench/checkpoints}"

DATA_ROOT="${DATA_ROOT:-${DATA_BASE}/control_plus_ifn/unseen_diff_moa}"
LOG_ROOT="${LOG_ROOT:-${LOGDIR}/fig2/task1_unseenMOA/diff}"
SAMPLES_ROOT="${SAMPLES_ROOT:-${SAMPLES_BASE}/fig2/task1_unseenMOA/diff}"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_BASE}/fig2/task1_unseenMOA/diff}"
mkdir -p "${LOG_ROOT}" "${SAMPLES_ROOT}" "${CKPT_ROOT}"

# -------------------- Discover datasets ----------------
# Convention 1: <dataset_base>_train__plus_control.h5ad / <dataset_base>_test__plus_control.h5ad
# Convention 2 (legacy): <dataset_base>_control_train.h5ad / <dataset_base>_control_test.h5ad
mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_train__plus_control.h5ad" | sort)
FILE_PATTERN="diff_moa"
if [[ ${#TRAIN_FILES[@]} -eq 0 ]]; then
  mapfile -t TRAIN_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name "*_control_train.h5ad" | sort)
  FILE_PATTERN="control_train"
fi
if [[ ${#TRAIN_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No *_train__plus_control.h5ad nor *_control_train.h5ad found under: ${DATA_ROOT}" >&2
  exit 1
fi

echo "Found ${#TRAIN_FILES[@]} datasets under ${DATA_ROOT}"
echo "Config: runs=${NUM_RUNS} | name=${NAME} | batch_size=${BATCH_SIZE} | num_workers=${NUM_WORKERS} | t_sample=${T_SAMPLE}"
echo

# -------------------- Main Loop ------------------------
for train_path in "${TRAIN_FILES[@]}"; do
  train_fname="$(basename "${train_path}")"
  if [[ "${FILE_PATTERN}" == "diff_moa" ]]; then
    dataset_base="${train_fname%_train__plus_control.h5ad}"
    test_fname="${dataset_base}_test__plus_control.h5ad"
    train_ds="${dataset_base}_train"
    test_ds="${dataset_base}_test"
  else
    dataset_base="${train_fname%_control_train.h5ad}"
    test_fname="${dataset_base}_control_test.h5ad"
    train_ds="${dataset_base}_control_train"
    test_ds="${dataset_base}_control_test"
  fi
  test_path="${DATA_ROOT}/${test_fname}"

  if [[ ! -f "${test_path}" ]]; then
    echo "[ERROR] Missing test file for dataset_base=${dataset_base}: ${test_path}" >&2
    exit 1
  fi

  echo "######################################################################"
  echo "###   Starting pipeline for dataset: ${train_ds}"
  echo "######################################################################"

  # Output layout: task/split/method/<dataset_base>/run{i}
  OUT_DIR="${SAMPLES_ROOT}/${dataset_base}/scdiff"
  CSV_DIR="${OUT_DIR}/metrics"
  SEED_CSV="${CSV_DIR}/metrics_${test_ds}.csv"
  mkdir -p "${OUT_DIR}" "${CSV_DIR}"

  echo "== $(date '+%F %T') | dataset=${dataset_base} runs=${NUM_RUNS} =="

  # scDiff data config
  data_settings=()
  data_settings+=("data.params.batch_size=${BATCH_SIZE}")
  data_settings+=("data.params.num_workers=${NUM_WORKERS}")
  data_settings+=("model.params.t_sample=${T_SAMPLE}")
  data_settings+=("model.params.path_to_save_fig=")

  {
    # Use temp logs per run to avoid truncation of evaluation metrics
    RUN_LOGS=()
    for (( i=1; i<=NUM_RUNS; i++ )); do
      RUN_LOGS+=("${OUT_DIR}/.run${i}.log")
    done

    # -------- Run 1..NUM_RUNS --------
    for (( i=1; i<=NUM_RUNS; i++ )); do
      run_tag="run${i}"
      run_postfix="perturbation_${NAME}_${run_tag}"
      run_log="${RUN_LOGS[i-1]}"

      echo -e "\n--- Running ${run_tag} / ${NUM_RUNS} for ${train_ds} ---"

      python src/scDiff/main.py \
        --custom_data_path "${DATA_ROOT}" \
        --base configs/scdiff/eval_perturbation.yaml \
        --name "${NAME}" \
        --logdir "${LOGDIR}" \
        --postfix "${run_postfix}" \
        ${OFFLINE_SETTINGS} \
        data.params.train.target=scdiff.data.perturbation_drug.PerturbationDrugTrain \
        data.params.test.target=scdiff.data.perturbation_drug.PerturbationDrugTest \
        data.params.train.params.datadir="${DATA_ROOT}" \
        data.params.test.params.datadir="${DATA_ROOT}" \
        data.params.train.params.dataset=${dataset_base} \
        data.params.train.params.fname=${train_fname} \
        data.params.test.params.dataset=${dataset_base} \
        data.params.test.params.fname=${test_fname} \
        data.params.train.params.use_drug_cond=true \
        data.params.train.params.drug_key=perturbation \
        data.params.train.params.dose_key=dose_value \
        data.params.train.params.pert_key=perturbation_status \
        data.params.train.params.ctrl_key=Control \
        data.params.train.params.stim_key=IFN \
        data.params.train.params.allow_custom_dataset=true \
        data.params.train.params.celltype_key=celltype \
        data.params.train.params.highly_variable=false \
        data.params.test.params.use_drug_cond=true \
        data.params.test.params.drug_key=perturbation \
        data.params.test.params.dose_key=dose_value \
        data.params.test.params.pert_key=perturbation_status \
        data.params.test.params.ctrl_key=Control \
        data.params.test.params.stim_key=IFN \
        data.params.test.params.allow_custom_dataset=true \
        data.params.test.params.celltype_key=celltype \
        data.params.test.params.highly_variable=false \
        model.params.generation_kwargs.n_samples=1000 \
        "${data_settings[@]}" 2>&1 | tee "${run_log}" || true
    done

    echo
    # Parse metrics from run logs
    cat "${RUN_LOGS[@]}" 2>/dev/null | awk -v ds="${test_ds}" -v num_runs="${NUM_RUNS}" -v method="scDiff(${NAME})" -v csv_path="${SEED_CSV}" '
      BEGIN {
        c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=0;
        c_pearson_all=c_pearson_delta_all=c_pearson_delta_de20=c_pearson_delta_de50=c_pearson_delta_de100=0;
      }
      function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

      # Capture metrics (11)
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = to_num($NF) }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = to_num($NF) }
      /Differential Expression Score \(DES\):/    { des[c_des++] = to_num($NF) }
      /E-Distance:/                               { edist[c_edist++] = to_num($NF) }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = to_num($NF) }
      /R-squared \(R2\):/                         { r2[c_r2++] = to_num($NF) }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++] = to_num($NF) }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = to_num($NF) }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = to_num($NF) }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = to_num($NF) }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = to_num($NF) }

      function mean_std(idx,    i,n,s,mu,ss,v) {
        if (idx==1)  { n=c_pds;                 for(i=0;i<n;i++){v=pds[i];                 s+=v} }
        else if(idx==2){ n=c_mae;               for(i=0;i<n;i++){v=mae[i];                 s+=v} }
        else if(idx==3){ n=c_des;               for(i=0;i<n;i++){v=des[i];                 s+=v} }
        else if(idx==4){ n=c_edist;             for(i=0;i<n;i++){v=edist[i];               s+=v} }
        else if(idx==5){ n=c_mmd;               for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
        else if(idx==6){ n=c_r2;                for(i=0;i<n;i++){v=r2[i];                  s+=v} }
        else if(idx==7){ n=c_pearson_all;       for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
        else if(idx==8){ n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
        else if(idx==9){ n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
        else if(idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v} }
        else if(idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
        mu = (n>0)? s/n : 0;
        for(i=0;i<n;i++){
          if (idx==1) v=pds[i];
          else if(idx==2) v=mae[i];
          else if(idx==3) v=des[i];
          else if(idx==4) v=edist[i];
          else if(idx==5) v=mmd[i];
          else if(idx==6) v=r2[i];
          else if(idx==7) v=pearson_all[i];
          else if(idx==8) v=pearson_delta_all[i];
          else if(idx==9) v=pearson_delta_de20[i];
          else if(idx==10) v=pearson_delta_de50[i];
          else if(idx==11) v=pearson_delta_de100[i];
          ss += (v - mu) * (v - mu);
        }
        return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
      }

      function val(idx, j,    v){
        if (idx==1) v=pds[j];
        else if(idx==2) v=mae[j];
        else if(idx==3) v=des[j];
        else if(idx==4) v=edist[j];
        else if(idx==5) v=mmd[j];
        else if(idx==6) v=r2[j];
        else if(idx==7) v=pearson_all[j];
        else if(idx==8) v=pearson_delta_all[j];
        else if(idx==9) v=pearson_delta_de20[j];
        else if(idx==10) v=pearson_delta_de50[j];
        else if(idx==11) v=pearson_delta_de100[j];
        return v;
      }

      function print_stat(idx, name,    i,n,s,mu,ss,std,v) {
        if (idx==1)  { n=c_pds;                 for(i=0;i<n;i++){v=pds[i];                 s+=v} }
        else if(idx==2){ n=c_mae;               for(i=0;i<n;i++){v=mae[i];                 s+=v} }
        else if(idx==3){ n=c_des;               for(i=0;i<n;i++){v=des[i];                 s+=v} }
        else if(idx==4){ n=c_edist;             for(i=0;i<n;i++){v=edist[i];               s+=v} }
        else if(idx==5){ n=c_mmd;               for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
        else if(idx==6){ n=c_r2;                for(i=0;i<n;i++){v=r2[i];                  s+=v} }
        else if(idx==7){ n=c_pearson_all;       for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
        else if(idx==8){ n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
        else if(idx==9){ n=c_pearson_delta_de20;for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
        else if(idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v} }
        else if(idx==11){ n=c_pearson_delta_de100;for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
        else n=0;
        if (n > 0) {
          mu = s / n;
          ss = 0;
          for (i = 0; i < n; i++) {
            if (idx==1) v=pds[i]; else if(idx==2) v=mae[i]; else if(idx==3) v=des[i];
            else if(idx==4) v=edist[i]; else if(idx==5) v=mmd[i]; else if(idx==6) v=r2[i];
            else if(idx==7) v=pearson_all[i]; else if(idx==8) v=pearson_delta_all[i];
            else if(idx==9) v=pearson_delta_de20[i]; else if(idx==10) v=pearson_delta_de50[i];
            else if(idx==11) v=pearson_delta_de100[i];
            ss += (v - mu) * (v - mu);
          }
          std = (n > 1) ? sqrt(ss / (n - 1)) : 0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std;
        } else {
          printf "%-40s: N/A (No data)\n", name;
        }
      }

      END {
        print "==================================================================";
        printf " Final statistics for %s (%d runs: train+eval)\n", ds, num_runs;
        print "==================================================================";
        print_stat(1, "Perturbation Discrimination (PDS)");
        print_stat(2, "Mean Absolute Error (MAE)");
        print_stat(3, "Differential Expression Score (DES)");
        print "----------------------------------------";
        print_stat(4, "E-Distance");
        print_stat(5, "Maximum Mean Discrepancy (MMD)");
        print_stat(6, "R-squared (R2)");
        print "----------------------------------------";
        print_stat(7, "Pearson (all genes)");
        print_stat(8, "Pearson Delta (all genes)");
        print_stat(9, "Pearson Delta (top 20 DE genes)");
        print_stat(10, "Pearson Delta (top 50 DE genes)");
        print_stat(11, "Pearson Delta (top 100 DE genes)");
        print "==================================================================\n";

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

        header = "Method";
        for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
        for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];

        row = method;
        for (i=1;i<=11;i++) {
          ms = mean_std(i); split(ms, parts, "|");
          row = row sprintf(",%.4f±%.4f", parts[1], parts[2]);
        }
        for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));

        print header > csv_path;
        print row    >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    '

    rm -f "${RUN_LOGS[@]}"
    echo
    echo "--- Finished pipeline for dataset: ${train_ds} ---"
    echo
  }

done

echo "######################################################################"
echo "###   All datasets completed! CSVs are under ${SAMPLES_ROOT}/*/scdiff/metrics/metrics_*.csv"
echo "######################################################################"
