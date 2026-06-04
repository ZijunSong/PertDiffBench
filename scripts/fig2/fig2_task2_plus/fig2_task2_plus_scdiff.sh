#!/usr/bin/env bash
# scDiff for Fig2 task2+: leave-one-out x (p0 / p0.25 / p0.5). Pass each fold directory as --custom_data_path.
#
# Optional: export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0" (or "p0.25", "p0.5", or "p0 p0.25")
# to restrict control fractions. Wrappers fig2_task2_plus_scdiff_p0.sh etc. set this.
# Optional: FIG2_TASK2_PLUS_REVERSE_HOLDOUT=1 → iterate HOLDOUT_TYPES as NK … B (reverse of default).
# Optional: FIG2_TASK2_PLUS_HOLDOUT_TYPES="CD8T CD14+Mono" → only these holdouts (space-separated;
# names must match default list: B, CD4T, CD8T, CD14+Mono, Dendritic, FCGR3A+Mono, NK).
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# Default single-GPU (override with CUDA_VISIBLE_DEVICES); wrappers default to 0/1/2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
LOGDIR="${LOGDIR:-logs}"
NAME="${NAME:-v7.5}"
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OFFLINE_SETTINGS="--wandb f"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scDiff}

HOMEDIR="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"
cd "$HOMEDIR"
# Numba (dcor import) cache → /data/ppnm/cache/numba to avoid filling $HOME.
if [[ -f "${HOMEDIR}/scripts/env_numba_cache_data.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOMEDIR}/scripts/env_numba_cache_data.sh"
fi
# GEARS → dcor uses Numba; on some CPUs LLVM fails with "Symbol not found" for JIT'd gufuncs.
# Disable JIT (slightly slower distance-correlation) — override with NUMBA_DISABLE_JIT=0 if needed.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
echo "Current working directory: $(pwd)"

DATA_BASE="${DATA_BASE:-data/fig2/task2_unseen_celltype_plus}"
DATA_ROOT="${HOMEDIR}/${DATA_BASE}"

if [[ -n "${FIG2_TASK2_PLUS_HOLDOUT_TYPES:-}" ]]; then
  read -r -a HOLDOUT_TYPES <<< "${FIG2_TASK2_PLUS_HOLDOUT_TYPES}"
  echo "[fig2_task2_plus_scdiff] HOLDOUT_TYPES (override): ${HOLDOUT_TYPES[*]}"
else
  HOLDOUT_TYPES=( "B" "CD4T" "CD8T" "CD14+Mono" "Dendritic" "FCGR3A+Mono" "NK" )
fi
if [[ "${FIG2_TASK2_PLUS_REVERSE_HOLDOUT:-}" == "1" ]]; then
  _rev_ht=()
  for ((_j=${#HOLDOUT_TYPES[@]}-1; _j>=0; _j--)); do
    _rev_ht+=("${HOLDOUT_TYPES[_j]}")
  done
  HOLDOUT_TYPES=("${_rev_ht[@]}")
  echo "[fig2_task2_plus_scdiff] HOLDOUT_TYPES (reversed): ${HOLDOUT_TYPES[*]}"
fi
if [[ -n "${FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS:-}" ]]; then
  read -r -a CTRL_SLUGS <<< "${FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS}"
else
  CTRL_SLUGS=( "p0" "p0.25" "p0.5" )
fi

LOG_ROOT="${LOGDIR}/fig2/fig2_task2_plus_scdiff"
SAMPLES_ROOT="samples/fig2/fig2_task2_plus/scdiff"
CKPT_ROOT="checkpoints/fig2/fig2_task2_plus/scdiff"
mkdir -p "${LOG_ROOT}" "${SAMPLES_ROOT}" "${CKPT_ROOT}"

TRAIN_FNAME="task2_train_exp.h5ad"
TEST_FNAME="task2_test_exp.h5ad"

for ht in "${HOLDOUT_TYPES[@]}"; do
  dataset_base="task2_loo_${ht}"

  for slug in "${CTRL_SLUGS[@]}"; do
    DATA_DIR="${DATA_ROOT}/loo_${ht}/${slug}"
    ds_tag="${ht}_${slug}"

    if [[ ! -f "${DATA_DIR}/${TRAIN_FNAME}" || ! -f "${DATA_DIR}/${TEST_FNAME}" ]]; then
      echo "[WARN] Skip ${ds_tag}: missing data under ${DATA_DIR}"
      continue
    fi

    echo "######################################################################"
    echo "###   scDiff | ${ds_tag} | dataset=${dataset_base}"
    echo "######################################################################"

    OUT_DIR="${SAMPLES_ROOT}/${ds_tag}"
    SEED_CSV="${OUT_DIR}/metrics_${ds_tag}.csv"
    mkdir -p "${OUT_DIR}"

    data_settings=()
    data_settings+=("data.params.train.params.dataset=${dataset_base}")
    data_settings+=("data.params.train.params.fname=${TRAIN_FNAME}")
    data_settings+=("data.params.test.params.dataset=${dataset_base}")
    data_settings+=("data.params.test.params.fname=${TEST_FNAME}")
    data_settings+=("model.params.generation_kwargs.n_samples=1000")

    {
      all_outputs=""

      for (( i=1; i<=NUM_RUNS; i++ )); do
        run_tag="run${i}"
        run_postfix="perturbation_${NAME}_${ds_tag}_${run_tag}"

        echo -e "\n--- Running ${run_tag} / ${NUM_RUNS} for ${ds_tag} ---"

        output=$(
          python src/scDiff/main.py \
            --custom_data_path "${DATA_DIR}" \
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
      echo -e "$all_outputs" | awk -v ds="${ds_tag}" -v num_runs="${NUM_RUNS}" -v method="scDiff(${NAME})" -v csv_path="${SEED_CSV}" '
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
      echo "--- Finished scDiff for ${ds_tag} ---"
      echo
    }

  done
done

echo "######################################################################"
echo "###   fig2_task2_plus scDiff finished. CSVs: ${SAMPLES_ROOT}/*/metrics_*.csv"
echo "######################################################################"
