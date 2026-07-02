#!/bin/bash
# 3 train+eval runs; aggregate to CSV

# Exit immediately on command failure.
trap "echo ERROR && exit 1" ERR
set -e

# --------------------
# (onlypath )
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=${NAME:-v7.5}
OFFLINE_SETTINGS="${OFFLINE_SETTINGS:---wandb_offline t}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiff}"          # write to CSV method name in
METHOD_DIR="${METHOD_DIR:-scdiff}" # usingin samples directory after level
# --------------------

# data root ( )
DATA_ROOT="data/add_zero_inflation_output"

# HOMEDIR asrepo root, Assume subdir .
HOMEDIR="$(dirname "$(dirname "$(realpath "$0")")")/../.."
cd "$HOMEDIR"
echo "current directory: $(pwd)"

# log root ( log)
RUNLOG_ROOT="${LOGDIR}/perturbation_${NAME}"
mkdir -p "${RUNLOG_ROOT}"

# definemusthandleallcelltypearray
CELL_TYPES=('CD4T')

NOISE_LEVELS=('0.3' '0.6' '1.0' '1.6' '2.2')

# outer loop: loop cell types
for cell_type in "${CELL_TYPES[@]}"; do
  # middle loop: loop noise levels
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###  Processing: $cell_type | noise: $noise_level"
    echo "######################################################################"

    # build noisy train path/validatedatafilename (path to DATA_ROOT)
    train_fname="task1_train_${cell_type}_exp_zeroinflation_strength_${noise_level}.h5ad"
    valid_fname="task1_valid_${cell_type}_exp_zeroinflation_strength_${noise_level}.h5ad"

    # checktrainfilewhetherexist
    if [ ! -f "${DATA_ROOT}/${train_fname}" ]; then
      echo " : Training data file not found '${DATA_ROOT}/${train_fname}'.skip ."
      continue
    fi

    # data (onlypathname )
    dataset_name="fig1_task1_${cell_type}_noise_${noise_level}"
    data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${train_fname}"
    data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${valid_fname}"

    # level andoutput dir
    COMBO_TAG="${cell_type}_noise_${noise_level}"
    DATASET_LOG="${RUNLOG_ROOT}/${COMBO_TAG}.log"
    output_suffix="${cell_type}_noise_${noise_level}"

    # : samples path after leveldirectorynameusing METHOD_DIR 
    OUTDIR_BASE="samples/zero_inflation_technoise/${output_suffix}/${METHOD_DIR}"
    mkdir -p "${OUTDIR_BASE}"

    echo -e "\n==== $(date '+%F %T') | Begin ${COMBO_TAG} ====\n" | tee -a "${DATASET_LOG}"

    # evaloutput
    all_outputs=""

    # inside : (each runcall main.py = train+eval)
    for (( i=1; i<=NUM_RUNS; i++ )); do
      export RUN_SEED=$(($i-1))
      echo -e "\n======================"                 | tee -a "${DATASET_LOG}"
      echo -e " Run ${i}/${NUM_RUNS} : ${COMBO_TAG}"    | tee -a "${DATASET_LOG}"
      echo -e "======================"                 | tee -a "${DATASET_LOG}"

      # only --custom_data_path to DATA_ROOT; args call 
      output=$(python src/scDiff/main.py \
        --custom_data_path "${DATA_ROOT}" \
        --base configs/scdiff/eval_perturbation.yaml \
        --name "${NAME}_${COMBO_TAG}_run${i}" \
        --logdir "${LOGDIR}" \
        --postfix "perturbation_${NAME}" \
        ${OFFLINE_SETTINGS} \
        ${data_settings} 2>&1) || true

      # print and log
      echo "${output}" | tee -a "${DATASET_LOG}"

      # append to stats text (keep )
      all_outputs+="${output}"$'\n'
    done

    # ==== statsto + CSV ====
    CSV_FILE="${OUTDIR_BASE}/metrics_${METHOD_DIR}_${COMBO_TAG}.csv"
    mkdir -p "$(dirname "${CSV_FILE}")"

    echo -e "\n" | tee -a "${DATASET_LOG}"
    echo -e "${all_outputs}" | awk \
      -v method="${METHOD_NAME}" \
      -v num_runs="${NUM_RUNS}" \
      -v csv_path="${CSV_FILE}" \
      -v combo="${COMBO_TAG}" \
      -v ds="${cell_type}" \
      -v nz="${noise_level}" '
      BEGIN{
        c_pds=c_mae=c_des=c_edist=c_mmd=c_r2=c_pearson_all=c_pearson_delta_all=c_pearson_delta_de20=c_pearson_delta_de50=c_pearson_delta_de100=0
      }
      function to_num(x){ gsub(/[^0-9eE+\-\.]/,"",x); return x+0 }

      # -------- 11 (from countvalue)--------
      /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++]                   = to_num($NF); next }
      /Mean Absolute Error \(MAE\):/              { mae[c_mae++]                   = to_num($NF); next }
      /Differential Expression Score \(DES\):/    { des[c_des++]                   = to_num($NF); next }
      /E-Distance:/                               { edist[c_edist++]               = to_num($NF); next }
      /Maximum Mean Discrepancy \(MMD\):/         { mmd[c_mmd++]                   = to_num($NF); next }
      /R-squared \(R2\):/                         { r2[c_r2++]                     = to_num($NF); next }
      /Pearson \(all genes\):/                    { pearson_all[c_pearson_all++]   = to_num($NF); next }
      /Pearson Delta \(all genes\):/              { pearson_delta_all[c_pearson_delta_all++] = to_num($NF); next }
      /Pearson Delta \(top 20 DE genes\):/        { pearson_delta_de20[c_pearson_delta_de20++] = to_num($NF); next }
      /Pearson Delta \(top 50 DE genes\):/        { pearson_delta_de50[c_pearson_delta_de50++] = to_num($NF); next }
      /Pearson Delta \(top 100 DE genes\):/       { pearson_delta_de100[c_pearson_delta_de100++] = to_num($NF); next }

      function mean(a,n, s,i){ s=0; for(i=0;i<n;i++) s+=a[i]; return n? s/n : 0 }
      function std(a,n,  mu,s,i){ if(n<=1) return 0; mu=mean(a,n); s=0; for(i=0;i<n;i++) s+=(a[i]-mu)*(a[i]-mu); return sqrt(s/(n-1)) }

      function mean_std(idx,  n,mu,sd){
        if(idx==1){ n=c_pds;                  mu=mean(pds,n);                  sd=std(pds,n) }
        else if(idx==2){ n=c_mae;            mu=mean(mae,n);                  sd=std(mae,n) }
        else if(idx==3){ n=c_des;            mu=mean(des,n);                  sd=std(des,n) }
        else if(idx==4){ n=c_edist;          mu=mean(edist,n);                sd=std(edist,n) }
        else if(idx==5){ n=c_mmd;            mu=mean(mmd,n);                  sd=std(mmd,n) }
        else if(idx==6){ n=c_r2;             mu=mean(r2,n);                   sd=std(r2,n) }
        else if(idx==7){ n=c_pearson_all;    mu=mean(pearson_all,n);          sd=std(pearson_all,n) }
        else if(idx==8){ n=c_pearson_delta_all;   mu=mean(pearson_delta_all,n);   sd=std(pearson_delta_all,n) }
        else if(idx==9){ n=c_pearson_delta_de20;  mu=mean(pearson_delta_de20,n);  sd=std(pearson_delta_de20,n) }
        else if(idx==10){n=c_pearson_delta_de50;  mu=mean(pearson_delta_de50,n);  sd=std(pearson_delta_de50,n) }
        else if(idx==11){n=c_pearson_delta_de100; mu=mean(pearson_delta_de100,n); sd=std(pearson_delta_de100,n) }
        return sprintf("%.6f±%.6f", mu, sd)
      }

      function val(idx, j, v){
        if      (idx==1)  v=pds[j];
        else if (idx==2)  v=mae[j];
        else if (idx==3)  v=des[j];
        else if (idx==4)  v=edist[j];
        else if (idx==5)  v=mmd[j];
        else if (idx==6)  v=r2[j];
        else if (idx==7)  v=pearson_all[j];
        else if (idx==8)  v=pearson_delta_all[j];
        else if (idx==9)  v=pearson_delta_de20[j];
        else if (idx==10) v=pearson_delta_de50[j];
        else if (idx==11) v=pearson_delta_de100[j];
        return (v=="") ? 0 : v;
      }

      function print_stat(name, data, count,   i,s,mu,ss,std_val){
        if (count>0){
          for(i=0;i<count;i++) s+=data[i];
          mu=s/count;
          for(i=0;i<count;i++) ss+=(data[i]-mu)*(data[i]-mu);
          std_val=(count>1)?sqrt(ss/(count-1)):0;
          printf "%-40s: %.4f ± %.4f\n", name, mu, std_val;
        } else {
          printf "%-40s: N/A (no data collected)\n", name;
        }
      }

      END {
        print "==================================================================";
        printf " stats: %s (%d )\n", combo, num_runs;
        print "==================================================================";

        print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
        print_stat("Mean Absolute Error (MAE)",         mae, c_mae);
        print_stat("Differential Expression Score (DES)", des, c_des);
        print "----------------------------------------";
        print_stat("E-Distance",                        edist, c_edist);
        print_stat("Maximum Mean Discrepancy (MMD)",    mmd,  c_mmd);
        print_stat("R-squared (R2)",                    r2,   c_r2);
        print "----------------------------------------";
        print_stat("Pearson (all genes)",               pearson_all,         c_pearson_all);
        print_stat("Pearson Delta (all genes)",         pearson_delta_all,   c_pearson_delta_all);
        print_stat("Pearson Delta (top 20 DE genes)",   pearson_delta_de20,  c_pearson_delta_de20);
        print_stat("Pearson Delta (top 50 DE genes)",   pearson_delta_de50,  c_pearson_delta_de50);
        print_stat("Pearson Delta (top 100 DE genes)",  pearson_delta_de100, c_pearson_delta_de100);
        print "==================================================================\n";

        # ---------- CSV: Dataset, Noise, Method + mean±std + per run ----------
        metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES";
        metric_names[4]="E-Distance"; metric_names[5]="MMD"; metric_names[6]="R2";
        metric_names[7]="Pearson (all genes)";
        metric_names[8]="Pearson Delta (all genes)";
        metric_names[9]="Pearson Delta (top 20 DE genes)";
        metric_names[10]="Pearson Delta (top 50 DE genes)";
        metric_names[11]="Pearson Delta (top 100 DE genes)";

        header="Dataset,Noise,Method";
        for(i=1;i<=11;i++) header=header "," metric_names[i] " (mean±std)";
        for(r=1;r<=num_runs;r++) for(i=1;i<=11;i++) header=header ",Run" r " " metric_names[i];

        row=ds "," nz "," method;
        for(i=1;i<=11;i++){
          ms=mean_std(i); split(ms, parts, "|");
          row=row sprintf(",%.6f±%.6f", parts[1], parts[2]);
        }
        for(r=0;r<num_runs;r++) for(i=1;i<=11;i++) row=row sprintf(",%.6f", val(i,r)+0);

        print header > csv_path;
        print row    >> csv_path;
        close(csv_path);
        printf("CSV written: %s\n", csv_path);
      }
    ' | tee -a "${DATASET_LOG}"

    echo -e "\n--- done : ${COMBO_TAG} ---\n" | tee -a "${DATASET_LOG}"
  done
done

echo "######################################################################"
echo "###   All cell types and noise levels finished!                 ###"
echo "######################################################################"
