#!/bin/bash
# Fig4 时间条件生成 — scDiffusion（设定一：训练 0h/2h/8h/10h，生成 4h/6h 与真实比较）

set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NUM_GENES="${NUM_GENES:-3000}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME="${METHOD_NAME:-scDiffusion}"
N_SAMPLES="${N_SAMPLES:-500}"

DATA_FIG4="/data/ppnm/data/PertDiffBench/data/fig4_task1"
TRAIN_H5="${DATA_FIG4}/fig4_train.h5ad"
TEST_H5="${DATA_FIG4}/fig4_test.h5ad"
# 预训练 VAE（encoder.ckpt / decoder.ckpt / gene_order.tsv），与 fig2 一致
VAE_STATE_DICT="${VAE_STATE_DICT:-checkpoints/annotation_model_v1}"

mkdir -p "${LOGDIR}/fig4_task1"
vae_base="checkpoints/scdiffusion/vae_checkpoint/fig4_${NUM_GENES}"
diff_base="checkpoints/scdiffusion/diffusion_checkpoint/fig4_${NUM_GENES}"
cls_base="checkpoints/scdiffusion/classifier_checkpoint/fig4_${NUM_GENES}"
sample_base="samples/fig4/scDiffusion_${NUM_GENES}"
csv_path="${sample_base}/metrics_${METHOD_NAME}_fig4_hvg_${NUM_GENES}.csv"
log_file="${LOGDIR}/fig4_task1/scdiffusion_fig4_hvg_${NUM_GENES}.log"
mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_base}"

# Label encoder from classifier dir (0h,2h,8h,10h -> indices 0,1,2,3)
CLS_DIR="${cls_base}/run1"
LABEL_ENC="${CLS_DIR}/label_encoder.npz"

{
  echo "== $(date '+%F %T') | fig4 | genes=${NUM_GENES} runs=${NUM_RUNS} n_samples=${N_SAMPLES} =="
  all_outputs=""

  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo "====================== Run ${i}/${NUM_RUNS} ======================"
    vae_dir="${vae_base}/run${i}"
    diff_dir="${diff_base}/run${i}"
    cls_dir="${cls_base}/run${i}"
    run_sample_dir="${sample_base}/run${i}"
    mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${run_sample_dir}"

    vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
    diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
    cls_ckpt="${cls_dir}/model009999.pt"

    # Step 1: Train VAE (pretrained from VAE_STATE_DICT: encoder.ckpt, decoder.ckpt, gene_order.tsv)
    echo "--- Step 1: Training VAE [run ${i}] ---"
    pushd src/scDiffusion/VAE >/dev/null
    python VAE_train.py --data_dir "../../../${TRAIN_H5}" --num_genes "${NUM_GENES}" --save_dir "../../../${vae_dir}" --state_dict "../../../${VAE_STATE_DICT}" || true
    popd >/dev/null

    # Step 2: Train Diffusion
    echo "--- Step 2: Training Diffusion [run ${i}] ---"
    pushd src/scDiffusion >/dev/null
    python cell_train.py --data_dir "../../${TRAIN_H5}" --vae_path "../../${vae_ckpt}" --save_dir "../../${diff_dir}" || true
    popd >/dev/null

    # Step 3: Train Classifier (time as label)
    echo "--- Step 3: Training Classifier (label_key=treatment_time) [run ${i}] ---"
    pushd src/scDiffusion >/dev/null
    python classifier_train.py --data_dir "../../${TRAIN_H5}" --vae_path "../../${vae_ckpt}" --model_path "../../${cls_dir}" --label_key treatment_time || true
    popd >/dev/null

    # Step 4: Generate 4h and 6h via classifier gradient interpolation (2h–8h → 4h/6h, 图2); then eval
    echo "--- Step 4: Sampling 4h & 6h (gradient interp) + Eval [run ${i}] ---"
    pushd src/scDiffusion >/dev/null
    # 4h: interp 2h(1) and 8h(2) with weight 5,5
    python classifier_sample.py \
      --num_samples "${N_SAMPLES}" --train-data-path "../../${TRAIN_H5}" --model_path "../../${diff_ckpt}" \
      --classifier_path "../../${cls_ckpt}" --ae_dir "../../${vae_ckpt}" --num_gene "${NUM_GENES}" \
      --sample_dir "../../${run_sample_dir}/4h" --out_h5ad "../../${run_sample_dir}/synthetic_4h.h5ad" \
      --init_cell_path "../../${TRAIN_H5}" --label_key treatment_time --label_encoder_path "../../${cls_dir}/label_encoder.npz" \
      --cell_type 1 2 --weight 5 5 --target_time_label 4h 2>&1 || true
    # 6h: interp weight 2.5, 7.5
    python classifier_sample.py \
      --num_samples "${N_SAMPLES}" --train-data-path "../../${TRAIN_H5}" --model_path "../../${diff_ckpt}" \
      --classifier_path "../../${cls_ckpt}" --ae_dir "../../${vae_ckpt}" --num_gene "${NUM_GENES}" \
      --sample_dir "../../${run_sample_dir}/6h" --out_h5ad "../../${run_sample_dir}/synthetic_6h.h5ad" \
      --init_cell_path "../../${TRAIN_H5}" --label_key treatment_time --label_encoder_path "../../${cls_dir}/label_encoder.npz" \
      --cell_type 1 2 --weight 2.5 7.5 --target_time_label 6h 2>&1 || true
    popd >/dev/null

    # Merge 4h and 6h into one h5ad for fig4 eval
    python -c "
import scanpy as sc
import os
r='${run_sample_dir}'
a4=sc.read_h5ad(os.path.join(r,'synthetic_4h.h5ad'))
a6=sc.read_h5ad(os.path.join(r,'synthetic_6h.h5ad'))
merged=sc.concat([a4,a6], join='inner')
merged.write_h5ad(os.path.join(r,'synthetic_fig4.h5ad'))
" 2>/dev/null || true

    output=$(python scripts/fig4/eval_fig4_time_conditioned.py \
      --test-h5ad "${TEST_H5}" --generated-h5ad "${run_sample_dir}/synthetic_fig4.h5ad" \
      --train-h5ad "${TRAIN_H5}" --n-samples "${N_SAMPLES}" 2>&1) || true
    echo "$output"
    all_outputs+="$output\n"
  done

  # Aggregate and write CSV (same AWK as fig1)
  echo -e "$all_outputs" | awk -v dataset="fig4" -v num_runs="${NUM_RUNS}" -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
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
      if (idx==1)  { n=c_pds; for(i=0;i<n;i++){v=pds[i]; s+=v} }
      else if(idx==2){ n=c_mae; for(i=0;i<n;i++){v=mae[i]; s+=v} }
      else if(idx==3){ n=c_des; for(i=0;i<n;i++){v=des[i]; s+=v} }
      else if(idx==4){ n=c_edist; for(i=0;i<n;i++){v=edist[i]; s+=v} }
      else if(idx==5){ n=c_mmd; for(i=0;i<n;i++){v=mmd[i]; s+=v} }
      else if(idx==6){ n=c_r2; for(i=0;i<n;i++){v=r2[i]; s+=v} }
      else if(idx==7){ n=c_pearson_all; for(i=0;i<n;i++){v=pearson_all[i]; s+=v} }
      else if(idx==8){ n=c_pearson_delta_all; for(i=0;i<n;i++){v=pearson_delta_all[i]; s+=v} }
      else if(idx==9){ n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i]; s+=v} }
      else if(idx==10){ n=c_pearson_delta_de50; for(i=0;i<n;i++){v=pearson_delta_de50[i]; s+=v} }
      else if(idx==11){ n=c_pearson_delta_de100; for(i=0;i<n;i++){v=pearson_delta_de100[i]; s+=v} }
      mu = (n>0)? s/n : 0;
      for(i=0;i<n;i++){
        if (idx==1) v=pds[i]; else if(idx==2) v=mae[i]; else if(idx==3) v=des[i]; else if(idx==4) v=edist[i];
        else if(idx==5) v=mmd[i]; else if(idx==6) v=r2[i]; else if(idx==7) v=pearson_all[i];
        else if(idx==8) v=pearson_delta_all[i]; else if(idx==9) v=pearson_delta_de20[i];
        else if(idx==10) v=pearson_delta_de50[i]; else if(idx==11) v=pearson_delta_de100[i];
        ss += (v - mu) * (v - mu);
      }
      return (n>1)? mu "|" sqrt(ss/(n-1)) : mu "|0";
    }
    function val(idx, j,    v){
      if (idx==1) v=pds[j]; else if(idx==2) v=mae[j]; else if(idx==3) v=des[j]; else if(idx==4) v=edist[j];
      else if(idx==5) v=mmd[j]; else if(idx==6) v=r2[j]; else if(idx==7) v=pearson_all[j];
      else if(idx==8) v=pearson_delta_all[j]; else if(idx==9) v=pearson_delta_de20[j];
      else if(idx==10) v=pearson_delta_de50[j]; else if(idx==11) v=pearson_delta_de100[j];
      return v;
    }
    END {
      print "==================================================================";
      printf " Final statistics for %s (%d runs)\n", dataset, num_runs;
      print "==================================================================";
      metric_names[1]="PDS"; metric_names[2]="MAE"; metric_names[3]="DES"; metric_names[4]="E-Distance";
      metric_names[5]="MMD"; metric_names[6]="R2"; metric_names[7]="Pearson (all genes)";
      metric_names[8]="Pearson Delta (all genes)"; metric_names[9]="Pearson Delta (top 20 DE genes)";
      metric_names[10]="Pearson Delta (top 50 DE genes)"; metric_names[11]="Pearson Delta (top 100 DE genes)";
      for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); printf "%-40s: %.4f ± %.4f\n", metric_names[i], parts[1], parts[2]; }
      header = "Method"; for (i=1;i<=11;i++) header = header "," metric_names[i] " (mean±std)";
      for (r=1;r<=num_runs;r++) for (i=1;i<=11;i++) header = header ",Run" r " " metric_names[i];
      row = method; for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
      for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));
      print header > csv_path; print row >> csv_path; close(csv_path);
      printf("CSV written: %s\n", csv_path);
    }
  '
  echo "--- Finished fig4 scDiffusion ---"
} 2>&1 | tee -a "${log_file}"

echo "######################################################################"
echo "###   fig4_task1 scDiffusion complete!                             ###"
echo "######################################################################"
