#!/bin/bash
# CellFM+DDPM pipeline: before scRNA -> CellFM encoder -> DDPM -> after scRNA
# using : in PertBench directory bash scripts/encoder_exp/cellfm/cellfm_ddpm.sh
set -e

# to directory ( in scripts/encoder_exp/cellfm/, needon level)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# in cellfm/cellfm_cuda11 (e.g. nohup conda), 
_env_name=""
if [ -n "${CONDA_PREFIX}" ]; then
  _env_name="$(basename "${CONDA_PREFIX}" 2>/dev/null)"
fi
if [ -z "${CONDA_PREFIX}" ] || [ "$_env_name" != "cellfm" ] && [ "$_env_name" != "cellfm_cuda11" ]; then
  for _conda_root in /opt/mamba /opt/conda "${HOME}/miniconda3" "${HOME}/anaconda3"; do
    if [ -f "${_conda_root}/etc/profile.d/conda.sh" ]; then
      set +e
      source "${_conda_root}/etc/profile.d/conda.sh" 2>/dev/null
      conda activate cellfm_cuda11 2>/dev/null || conda activate cellfm 2>/dev/null
      set -e
      [ -n "${CONDA_PREFIX}" ] && break
    fi
  done
fi

export HF_ENDPOINT=https://hf-mirror.com
# MindSpore (ERROR level )
export GLOG_minloglevel=2
export MS_LOG_LEVEL=3 2>/dev/null || true

# MindSpore GPU need found libcuda.so / libcudnn.so / libcublas.so
# sudo whencanusing conda CUDA/cuDNN, in $CONDA_PREFIX/lib, using
# : conda lib in front, to CUDA 13 (MindSpore must CUDA 11)
# LD_LIBRARY_PATH, Ensure conda lib in front
_ld_paths=()
if [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  _ld_paths+=("${CONDA_PREFIX}/lib")
fi
if [ -d /usr/lib/x86_64-linux-gnu ]; then
  _ld_paths+=("/usr/lib/x86_64-linux-gnu")
fi
# only conda CUDA when CUDA, to libcublas.so.12/13
if [ -z "${CONDA_PREFIX}" ] || [ ! -d "${CONDA_PREFIX}/lib" ]; then
  if [ -d /usr/local/cuda/lib64 ]; then
    _ld_paths+=("/usr/local/cuda/lib64")
  fi
fi
# LD_LIBRARY_PATH: path , afteradd path ( CUDA containpath)
if [ ${#_ld_paths[@]} -gt 0 ]; then
  _new_ld="$(IFS=:; echo "${_ld_paths[*]}")"
  _final_ld="${_new_ld}"
  if [ -n "${LD_LIBRARY_PATH}" ]; then
    # from LD_LIBRARY_PATH in pathcols path
    _old_paths=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n')
    for _old_path in $_old_paths; do
      # skipemptypath, CUDA path, toand in pathcols path
      [ -z "${_old_path}" ] && continue
      echo "${_new_ld}" | tr ':' '\n' | grep -q "^${_old_path}$" && continue
      echo "${_old_path}" | grep -q "^/usr/local/cuda" && continue
      _final_ld="${_final_ld}:${_old_path}"
    done
  fi
  export LD_LIBRARY_PATH="${_final_ld}"
  # EnsureLD_LIBRARY_PATHin canusing (MindSporeinimportwhencheck )
  export LD_LIBRARY_PATH
fi

CELL_TYPE="CD4T"
NUM_RUNS=3
METHOD_NAME="cellfm_ddpm"

# usingabsolute path, cwd tofile
TRAIN_H5="${REPO_ROOT}/data/fig1/raw_task1/task1_train_${CELL_TYPE}_exp.h5ad"
VALID_H5="${REPO_ROOT}/data/fig1/raw_task1/task1_valid_${CELL_TYPE}_exp.h5ad"
TRAIN_H5_WITH_LATENT="${REPO_ROOT}/samples/encoder_exp/cellfm_ddpm/task1_train_${CELL_TYPE}_with_cellfm_latent.h5ad"
VALID_H5_WITH_LATENT="${REPO_ROOT}/samples/encoder_exp/cellfm_ddpm/task1_valid_${CELL_TYPE}_with_cellfm_latent.h5ad"

CELLFM_CKPT_PATH="${CELLFM_CKPT_PATH:-${REPO_ROOT}/checkpoints/CellFM/CellFM_80M_weight.ckpt}"

LATENT_DDPM_BASE_DIR="${REPO_ROOT}/checkpoints/cellfm_ddpm/latent_ddpm"

EVAL_OUT_PREFIX="${REPO_ROOT}/samples/encoder_exp/cellfm_ddpm/cellfm_latent_ddpm_mlp_task1_${CELL_TYPE}_preds"
CSV_PATH="${REPO_ROOT}/samples/encoder_exp/cellfm_ddpm/metrics_${CELL_TYPE}.csv"

CONFIG_PATH="${REPO_ROOT}/configs/baselines/cellfm_ddpm_mlp.yaml"
LOG_DIR="${REPO_ROOT}/logs/cellfm_ddpm"

mkdir -p \
  "$(dirname "${TRAIN_H5_WITH_LATENT}")" \
  "$(dirname "${LATENT_DDPM_BASE_DIR}")" \
  "${LATENT_DDPM_BASE_DIR}" \
  "${LOG_DIR}"

echo "######################################################################"
echo "###   CellFM+DDPM pipeline for cell type: ${CELL_TYPE}"
echo "###   Requires MindSpore-GPU (CUDA/cuDNN, libcuda.so, libcudnn.so, libcublas.so in LD_LIBRARY_PATH)"
echo "######################################################################"

ALL_OUTPUTS=""

for (( run=1; run<=NUM_RUNS; run++ )); do
  echo
  echo "======================================================================"
  echo ">>> Run ${run}/${NUM_RUNS} for ${CELL_TYPE}"
  echo "======================================================================"

  RUN_DDPM_DIR="${LATENT_DDPM_BASE_DIR}/run_${run}"
  mkdir -p "${RUN_DDPM_DIR}"

  # inputdata exist directly 
  if [ ! -f "${TRAIN_H5}" ]; then
    echo "[ERROR] Train input not found: ${TRAIN_H5}"
    echo "Please prepare task1 data (see Readme) and re-run."
    exit 1
  fi
  if [ ! -f "${VALID_H5}" ]; then
    echo "[ERROR] Valid input not found: ${VALID_H5}"
    exit 1
  fi

  # filter MindSpore ( filter "Load dynamic library ... failed", to to )
  _filter_ms() {
    grep -v -e "libcuda.so.*not found" -e "libcudnn.so.*not found" \
             -e "device_target.*will be deprecated" \
             -e "parameters in the 'net' are not loaded" \
             -e "encoder\.[0-9]*\.attn\.q_proj\.weight" \
             -e "Load dynamic library: libmindspore_ascend" 2>/dev/null
  }

  echo -e "\n--- [1/4] Encode train_${CELL_TYPE} with CellFM (run ${run}) ---"
  # EnsureLD_LIBRARY_PATHinPython when (MindSporeinimportwhencheck )
  # usingenv Ensure amountin when 
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" python scripts/encoder_exp/cellfm/apply_cellfm_encoder.py \
    --data-path "${TRAIN_H5}" \
    --out-h5ad "${TRAIN_H5_WITH_LATENT}" \
    --ckpt-path "${CELLFM_CKPT_PATH}" \
    --device cuda 2>&1 | _filter_ms | tee "${LOG_DIR}/encode_train_${CELL_TYPE}_run_${run}.log"
  ENC_EXIT=${PIPESTATUS[0]}
  [ "${ENC_EXIT}" -ne 0 ] && { echo "[ERROR] [1/4] Encode train failed (exit ${ENC_EXIT}). Check ${LOG_DIR}/encode_train_${CELL_TYPE}_run_${run}.log"; exit 1; }

  echo -e "\n--- [2/4] Encode valid_${CELL_TYPE} with CellFM (run ${run}) ---"
  # EnsureLD_LIBRARY_PATHinPython when (MindSporeinimportwhencheck )
  # usingenv Ensure amountin when 
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" python scripts/encoder_exp/cellfm/apply_cellfm_encoder.py \
    --data-path "${VALID_H5}" \
    --out-h5ad "${VALID_H5_WITH_LATENT}" \
    --ckpt-path "${CELLFM_CKPT_PATH}" \
    --device cuda 2>&1 | _filter_ms | tee "${LOG_DIR}/encode_valid_${CELL_TYPE}_run_${run}.log"
  ENC_EXIT=${PIPESTATUS[0]}
  [ "${ENC_EXIT}" -ne 0 ] && { echo "[ERROR] [2/4] Encode valid failed (exit ${ENC_EXIT}). Check ${LOG_DIR}/encode_valid_${CELL_TYPE}_run_${run}.log"; exit 1; }

  if [ ! -f "${TRAIN_H5_WITH_LATENT}" ]; then
    echo "[ERROR] Encoded train h5ad not found: ${TRAIN_H5_WITH_LATENT}"
    echo "Step [1/4] may have failed without non-zero exit (e.g. pipe). Check ${LOG_DIR}/encode_train_${CELL_TYPE}_run_${run}.log"
    exit 1
  fi

  echo -e "\n--- [3/4] Train latent DDPM+decoder for ${CELL_TYPE} (run ${run}) ---"

  python scripts/encoder_exp/cellfm/train_cellfm_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    --train-data-path "${TRAIN_H5_WITH_LATENT}" \
    --save-weight-dir "${RUN_DDPM_DIR}" 2>&1 | tee "${LOG_DIR}/train_latent_ddpm_${CELL_TYPE}_run_${run}.log"

  echo -e "\n--- [4/4] Evaluate model on valid_${CELL_TYPE} (run ${run}) ---"

  CKPT_PATH="${RUN_DDPM_DIR}/model_final.pth"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "[WARN] Final checkpoint not found at ${CKPT_PATH}, skip eval for this run."
    continue
  fi

  OUTPUT=$(python scripts/encoder_exp/cellfm/eval_cellfm_latent_ddpm_mlp.py \
    -c "${CONFIG_PATH}" \
    -k "${CKPT_PATH}" \
    --data-path "${VALID_H5_WITH_LATENT}" \
    -n 200 \
    -o "${EVAL_OUT_PREFIX}_run_${run}.h5ad" 2>&1) || true

  echo "${OUTPUT}"
  ALL_OUTPUTS+="${OUTPUT}"$'\n'
done

echo
echo "--- Aggregating metrics to CSV: ${CSV_PATH} ---"
echo

echo "${ALL_OUTPUTS}" | awk -v dataset="${CELL_TYPE}" -v num_runs="${NUM_RUNS}" \
                       -v method="${METHOD_NAME}" -v csv_path="${CSV_PATH}" '
  /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
  /Mean Absolute Error \(MAE\):/             { mae[c_mae++] = $NF }
  /Differential Expression Score \(DES\):/   { des[c_des++] = $NF }
  /E-Distance:/                              { edist[c_edist++] = $NF }
  /Maximum Mean Discrepancy \(MMD\):/        { mmd[c_mmd++] = $NF }
  /R-squared \(R2\):/                        { r2[c_r2++] = $NF }
  /Pearson \(all genes\):/                   { pearson_all[c_pearson_all++] = $NF }
  /Pearson Delta \(all genes\):/             { pearson_delta_all[c_pearson_delta_all++] = $NF }
  /Pearson Delta \(top 20 DE genes\):/       { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
  /Pearson Delta \(top 50 DE genes\):/       { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
  /Pearson Delta \(top 100 DE genes\):/      { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

  function print_stat(name, arr, cnt,    i,sum,mean,ssd,sd){
    if (cnt > 0){
      sum=0
      for(i=0;i<cnt;i++) sum+=arr[i]
      mean=sum/cnt
      ssd=0
      for(i=0;i<cnt;i++) ssd+=(arr[i]-mean)^2
      sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
      printf "%-40s: %.4f ± %.4f\n", name, mean, sd
    } else {
      printf "%-40s: N/A (No data collected)\n", name
    }
  }

  function mean_std_str(idx,    i,sum,mean,ssd,sd,cnt){
    if(idx==1){ cnt=c_pds; for(i=0;i<cnt;i++) sum+=pds[i] }
    else if(idx==2){ cnt=c_mae; for(i=0;i<cnt;i++) sum+=mae[i] }
    else if(idx==3){ cnt=c_des; for(i=0;i<cnt;i++) sum+=des[i] }
    else if(idx==4){ cnt=c_edist; for(i=0;i<cnt;i++) sum+=edist[i] }
    else if(idx==5){ cnt=c_mmd; for(i=0;i<cnt;i++) sum+=mmd[i] }
    else if(idx==6){ cnt=c_r2; for(i=0;i<cnt;i++) sum+=r2[i] }
    else if(idx==7){ cnt=c_pearson_all; for(i=0;i<cnt;i++) sum+=pearson_all[i] }
    else if(idx==8){ cnt=c_pearson_delta_all; for(i=0;i<cnt;i++) sum+=pearson_delta_all[i] }
    else if(idx==9){ cnt=c_pearson_delta_de20; for(i=0;i<cnt;i++) sum+=pearson_delta_de20[i] }
    else if(idx==10){ cnt=c_pearson_delta_de50; for(i=0;i<cnt;i++) sum+=pearson_delta_de50[i] }
    else if(idx==11){ cnt=c_pearson_delta_de100; for(i=0;i<cnt;i++) sum+=pearson_delta_de100[i] }

    if(cnt>0){
      mean=sum/cnt; ssd=0
      if(idx==1){ for(i=0;i<cnt;i++) ssd+=(pds[i]-mean)^2 }
      else if(idx==2){ for(i=0;i<cnt;i++) ssd+=(mae[i]-mean)^2 }
      else if(idx==3){ for(i=0;i<cnt;i++) ssd+=(des[i]-mean)^2 }
      else if(idx==4){ for(i=0;i<cnt;i++) ssd+=(edist[i]-mean)^2 }
      else if(idx==5){ for(i=0;i<cnt;i++) ssd+=(mmd[i]-mean)^2 }
      else if(idx==6){ for(i=0;i<cnt;i++) ssd+=(r2[i]-mean)^2 }
      else if(idx==7){ for(i=0;i<cnt;i++) ssd+=(pearson_all[i]-mean)^2 }
      else if(idx==8){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_all[i]-mean)^2 }
      else if(idx==9){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de20[i]-mean)^2 }
      else if(idx==10){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de50[i]-mean)^2 }
      else if(idx==11){ for(i=0;i<cnt;i++) ssd+=(pearson_delta_de100[i]-mean)^2 }
      sd=(cnt>1)?sqrt(ssd/(cnt-1)):0
      return sprintf("%.4f|%.4f", mean, sd)
    }
    return "0.0000|0.0000"
  }

  function val_idx(idx, r,    v){
    if     (idx==1){ v = (r < c_pds)?pds[r]:"" }
    else if(idx==2){ v = (r < c_mae)?mae[r]:"" }
    else if(idx==3){ v = (r < c_des)?des[r]:"" }
    else if(idx==4){ v = (r < c_edist)?edist[r]:"" }
    else if(idx==5){ v = (r < c_mmd)?mmd[r]:"" }
    else if(idx==6){ v = (r < c_r2)?r2[r]:"" }
    else if(idx==7){ v = (r < c_pearson_all)?pearson_all[r]:"" }
    else if(idx==8){ v = (r < c_pearson_delta_all)?pearson_delta_all[r]:"" }
    else if(idx==9){ v = (r < c_pearson_delta_de20)?pearson_delta_de20[r]:"" }
    else if(idx==10){ v = (r < c_pearson_delta_de50)?pearson_delta_de50[r]:"" }
    else if(idx==11){ v = (r < c_pearson_delta_de100)?pearson_delta_de100[r]:"" }
    return v
  }

  END{
    print "=================================================================="
    printf " Final statistics for %s (%d runs)\n", dataset, num_runs
    print "=================================================================="

    print_stat("Perturbation Discrimination (PDS)", pds, c_pds)
    print_stat("Mean Absolute Error (MAE)", mae, c_mae)
    print_stat("Differential Expression Score (DES)", des, c_des)
    print "----------------------------------------"
    print_stat("E-Distance", edist, c_edist)
    print_stat("Maximum Mean Discrepancy (MMD)", mmd, c_mmd)
    print_stat("R-squared (R2)", r2, c_r2)
    print "----------------------------------------"
    print_stat("Pearson (all genes)", pearson_all, c_pearson_all)
    print_stat("Pearson Delta (all genes)", pearson_delta_all, c_pearson_delta_all)
    print_stat("Pearson Delta (top 20 DE genes)", pearson_delta_de20, c_pearson_delta_de20)
    print_stat("Pearson Delta (top 50 DE genes)", pearson_delta_de50, c_pearson_delta_de50)
    print_stat("Pearson Delta (top 100 DE genes)", pearson_delta_de100, c_pearson_delta_de100)

    print "==================================================================\n"

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

    header = "Method"
    for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)"
    for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i]

    row = method
    for (i=1;i<=11;i++){
      ms = mean_std_str(i)
      split(ms, parts, "|")
      row = row sprintf(",%.4f±%.4f", parts[1], parts[2])
    }
    for (r=0;r<num_runs;r++){
      for (i=1;i<=11;i++){
        v = val_idx(i, r)
        if (v == "") row = row ","
        else row = row sprintf(",%.4f", v)
      }
    }

    print header > csv_path
    print row    >> csv_path
    close(csv_path)
    printf("CSV written: %s\n", csv_path)
  }
'

echo "######################################################################"
echo "###   CellFM+DDPM pipeline for ${CELL_TYPE} complete!                ###"
echo "######################################################################"
