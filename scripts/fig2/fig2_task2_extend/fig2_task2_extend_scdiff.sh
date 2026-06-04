#!/usr/bin/env bash
# scDiff pipeline for task2 unseen cell type (scGen setting): train on CD4T, test on B and NK.
# data path as task2_unseen_celltype, logicand fig2_task1_scdiff.sh .
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# -------------------- Configuration --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="${LOGDIR:-logs}"
NAME="${NAME:-v7.5}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OFFLINE_SETTINGS="--wandb f"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scDiff}

# -------------------- Project Root ---------------------
HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"
echo "Current working directory: $(pwd)"

# -------------------- Paths ----------------------------
DATA_ROOT="data/fig2/task2_unseen_celltype"
LOG_ROOT="${LOGDIR}/fig2/task2_extend_scgen"
SAMPLES_ROOT="samples/fig2/task2_extend_scgen/scdiff"
CKPT_ROOT="checkpoints/fig2/task2_extend_scgen/scdiff"
mkdir -p "${LOG_ROOT}" "${SAMPLES_ROOT}" "${CKPT_ROOT}"

# -------------------- Datasets (task2: train=CD4T, test=B or NK) -------------------------
TARGET_CELL_TYPES=( "B" "NK" )
TRAIN_FNAME="task1_train_CD4T_exp.h5ad"

# -------------------- Main Loop ------------------------
for cell_type in "${TARGET_CELL_TYPES[@]}"; do
  dataset_base="task2_${cell_type}"
  test_fname="task2_test_${cell_type}_exp.h5ad"

  echo "######################################################################"
  echo "###   Starting pipeline for dataset: ${dataset_base} (train=CD4T, test=${cell_type})"
  echo "######################################################################"

  OUT_DIR="${SAMPLES_ROOT}/${dataset_base}"
  SEED_CSV="${OUT_DIR}/metrics_${dataset_base}.csv"
  mkdir -p "${OUT_DIR}"

  echo "== $(date '+%F %T') | dataset=${dataset_base} runs=${NUM_RUNS} =="

  # scDiff data : train using CD4T, test usingcurrent cell_type test set
  data_settings=()
  data_settings+=("data.params.train.params.dataset=${dataset_base}")
  data_settings+=("data.params.train.params.fname=${TRAIN_FNAME}")
  data_settings+=("data.params.test.params.dataset=${dataset_base}")
  data_settings+=("data.params.test.params.fname=${test_fname}")
  data_settings+=("model.params.generation_kwargs.n_samples=1000")

  {
    all_outputs=""

    for (( i=1; i<=NUM_RUNS; i++ )); do
      run_tag="run${i}"
      run_postfix="perturbation_${NAME}_${run_tag}"

      echo -e "\n--- Running ${run_tag} / ${NUM_RUNS} for ${dataset_base} ---"

      output=$(
        python src/scDiff/main.py \
          --custom_data_path "${DATA_ROOT}" \
          --base configs/scdiff/eval_perturbation.yaml \
          --name "${NAME}" \
          --logdir "${LOGDIR}" \
          --postfix "${run_postfix}" \
          ${OFFLINE_SETTINGS} \
          "${data_settings[@]}" 2>&1
      ) || true

      echo "$output"
      all_outputs+="$output"
      all_outputs+=$'\n'
    done

    echo
    echo -e "$all_outputs" | awk -v ds="${dataset_base}" -v num_runs="${NUM_RUNS}" -v method="scDiff(${NAME})" -v csv_path="${SEED_CSV}" '
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++] = $NF }
      /Differential Expression Score \(DES\):/    { des[c_des++] = $NF }
      /E-Distance:/                               { edist[c_edist++] = $NF }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++] = $NF }
      /R-squared \(R2\):/                         { r2[c_r2++] = $NF }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++] = $NF }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = $NF }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

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

      END {
        print "==================================================================";
        printf " Final statistics for %s (%d runs: train+eval)\n", ds, num_runs;
        print "==================================================================";
        metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES"; metric_names[4]="E-Distance";
        metric_names[5]="MMD"; metric_names[6]="R2"; metric_names[7]="Pearson (all genes)";
        metric_names[8]="Pearson Delta (all genes)"; metric_names[9]="Pearson Delta (top 20 DE genes)";
        metric_names[10]="Pearson Delta (top 50 DE genes)"; metric_names[11]="Pearson Delta (top 100 DE genes)";

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

    echo
    echo "--- Finished pipeline for dataset: ${dataset_base} ---"
    echo
  }

done

echo "######################################################################"
echo "###   All task2_extend scDiff runs completed!"
echo "###   CSVs are under ${SAMPLES_ROOT}/*/metrics_*.csv"
echo "######################################################################"
