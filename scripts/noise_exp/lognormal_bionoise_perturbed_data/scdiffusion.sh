#!/bin/bash

# Exit on error; print a clear message
set -e
trap 'echo "ERROR: a command failed. Exiting." >&2' ERR

# -------------------- Configuration --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NUM_GENES="${NUM_GENES:-6998}"
NUM_RUNS=${NUM_RUNS:-3}
METHOD_NAME=${METHOD_NAME:-scDiffusion}   # 仅作标注
# -------------------------------------------------------

# 细胞类型（可扩展）
CELL_TYPES=( 'CD4T' )

# 噪声标准差
NOISE_LEVELS=(0.1 0.25 0.5 1.0 1.5)

mkdir -p "${LOGDIR}/add_lognormal_bionoise_output"

for cell_type in "${CELL_TYPES[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "######################################################################"
    echo "###   Starting pipeline: cell=${cell_type} | noise=${noise_level}"
    echo "######################################################################"

    # -------------------- 数据路径（与你现有数据保持一致） --------------------
    train_h5="data/add_lognormal_bionoise_output/task1_train_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"
    valid_h5="data/add_lognormal_bionoise_output/task1_valid_${cell_type}_exp_lognorm_cv_${noise_level}.h5ad"

    # 缺失则跳过
    if [ ! -f "$train_h5" ]; then
      echo "警告: 未找到训练数据文件 '$train_h5'，跳过该组合。"
      continue
    fi

    # -------------------- 输出根目录（与范例目录风格一致） --------------------
    vae_base="checkpoints/scdiffusion/vae_checkpoint/lognormal_bionoise/${cell_type}_${NUM_GENES}_noise_${noise_level}"
    diff_base="checkpoints/scdiffusion/diffusion_checkpoint/lognormal_bionoise/${cell_type}_${NUM_GENES}_noise_${noise_level}"
    cls_base="checkpoints/scdiffusion/classifier_checkpoint/2-classifier/lognormal_bionoise/${cell_type}_${NUM_GENES}_noise_${noise_level}"
    sample_base="samples/lognormal_bionoise/${cell_type}/scDiffusion_${NUM_GENES}_noise_${noise_level}"
    mkdir -p "${vae_base}" "${diff_base}" "${cls_base}" "${sample_base}"

    # CSV 与日志（每 cell/noise 一份 CSV，便于横向比较）
    csv_path="${sample_base}/metrics_${METHOD_NAME}_${cell_type}_noise_${noise_level}_hvg_${NUM_GENES}.csv"
    log_file="${LOGDIR}/add_lognormal_bionoise_output/scdiffusion_${cell_type}_hvg_${NUM_GENES}_noise_${noise_level}.log"

    {
      echo "== $(date '+%F %T') | cell=${cell_type} genes=${NUM_GENES} runs=${NUM_RUNS} noise=${noise_level} =="

      all_outputs=""

      # -------------------- 3x（训练+测评） --------------------
      for (( i=1; i<=NUM_RUNS; i++ )); do
        echo
        echo "======================"
        echo " Run ${i}/${NUM_RUNS} | ${cell_type} | noise=${noise_level}"
        echo "======================"

        # 分 run 的目录
        vae_dir="${vae_base}/run${i}"
        diff_dir="${diff_base}/run${i}"
        cls_dir="${cls_base}/run${i}"
        run_sample_dir="${sample_base}/run${i}"
        mkdir -p "${vae_dir}" "${diff_dir}" "${cls_dir}" "${run_sample_dir}"

        # 约定的权重文件名（与范例一致）
        vae_ckpt="${vae_dir}/model_seed=0_step=9999.pt"
        diff_ckpt="${diff_dir}/my_diffusion/model010000.pt"
        cls_ckpt="${cls_dir}/model009999.pt"

        # --- Step 1: 训练 VAE ---
        echo
        echo "--- Step 1: Training VAE ---"
        pushd src/scDiffusion/VAE >/dev/null
        python VAE_train.py \
          --data_dir "../../../${train_h5}" \
          --num_genes "${NUM_GENES}" \
          --state_dict ../../../checkpoints/annotation_model_v1 \
          --save_dir "../../../${vae_dir}"
        popd >/dev/null

        # --- Step 2: 训练 Diffusion ---
        echo
        echo "--- Step 2: Training Diffusion ---"
        pushd src/scDiffusion >/dev/null
        python cell_train.py \
          --data_dir "../../${train_h5}" \
          --vae_path "../../${vae_ckpt}" \
          --save_dir "../../${diff_dir}"
        popd >/dev/null

        # --- Step 3: 训练分类器 ---
        echo
        echo "--- Step 3: Training Classifier ---"
        pushd src/scDiffusion >/dev/null
        python classifier_train.py \
          --data_dir "../../${train_h5}" \
          --vae_path "../../${vae_ckpt}" \
          --model_path "../../${cls_dir}"
        popd >/dev/null

        # --- Step 4: 采样与评估 ---
        echo
        echo "--- Step 4: Sampling & Evaluation ---"
        pushd src/scDiffusion >/dev/null
        output=$(
          python classifier_sample.py \
            --num_samples 8 \
            --train-data-path "../../${train_h5}" \
            --model_path "../../${diff_ckpt}" \
            --classifier_path "../../${cls_ckpt}" \
            --ae_dir "../../${vae_ckpt}" \
            --num_gene "${NUM_GENES}" \
            --sample_dir "../../${run_sample_dir}" \
            --out_h5ad "../../${run_sample_dir}/synthetic_ifn_${i}.h5ad" \
            --umap_plot "../../${run_sample_dir}/umap_comparison_${i}.png" \
            --init_cell_path "../../${valid_h5}" 2>&1
        ) || true
        popd >/dev/null

        echo "$output"
        all_outputs+="$output\n"
      done

      # -------------------- 统计 + 写 CSV（与范例一致） --------------------
      echo
      echo -e "$all_outputs" | awk -v dataset="$cell_type" -v noise="$noise_level" -v num_runs="$NUM_RUNS" -v method="${METHOD_NAME}" -v csv_path="${csv_path}" '
        # 捕获 11 个指标
        /Perturbation Discrimination Score \(PDS\):/ { pds[c_pds++] = $NF }
        /Mean Absolute Error \(MAE\):/               { mae[c_mae++] = $NF }
        /Differential Expression Score \(DES\):/     { des[c_des++] = $NF }
        /E-Distance:/                                { edist[c_edist++] = $NF }
        /Maximum Mean Discrepancy \(MMD\):/          { mmd[c_mmd++] = $NF }
        /R-squared \(R2\):/                          { r2[c_r2++] = $NF }
        /Pearson \(all genes\):/                     { pearson_all[c_pearson_all++] = $NF }
        /Pearson Delta \(all genes\):/               { pearson_delta_all[c_pearson_delta_all++] = $NF }
        /Pearson Delta \(top 20 DE genes\):/         { pearson_delta_de20[c_pearson_delta_de20++] = $NF }
        /Pearson Delta \(top 50 DE genes\):/         { pearson_delta_de50[c_pearson_delta_de50++] = $NF }
        /Pearson Delta \(top 100 DE genes\):/        { pearson_delta_de100[c_pearson_delta_de100++] = $NF }

        # mean|std 计算（按范例）
        function mean_std(idx,    i,n,s,mu,ss,v) {
          if (idx==1)  { n=c_pds;                  for(i=0;i<n;i++){v=pds[i];                 s+=v} }
          else if(idx==2){ n=c_mae;                for(i=0;i<n;i++){v=mae[i];                 s+=v} }
          else if(idx==3){ n=c_des;                for(i=0;i<n;i++){v=des[i];                 s+=v} }
          else if(idx==4){ n=c_edist;              for(i=0;i<n;i++){v=edist[i];               s+=v} }
          else if(idx==5){ n=c_mmd;                for(i=0;i<n;i++){v=mmd[i];                 s+=v} }
          else if(idx==6){ n=c_r2;                 for(i=0;i<n;i++){v=r2[i];                  s+=v} }
          else if(idx==7){ n=c_pearson_all;        for(i=0;i<n;i++){v=pearson_all[i];         s+=v} }
          else if(idx==8){ n=c_pearson_delta_all;  for(i=0;i<n;i++){v=pearson_delta_all[i];   s+=v} }
          else if(idx==9){ n=c_pearson_delta_de20; for(i=0;i<n;i++){v=pearson_delta_de20[i];  s+=v} }
          else if(idx==10){ n=c_pearson_delta_de50;for(i=0;i<n;i++){v=pearson_delta_de50[i];  s+=v} }
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

        # 取第 j 次 run 的原始值（0-based）
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

        function print_stat(name, data, count,    i,s,mu,ss,std) {
          if (count > 0) {
            for (i = 0; i < count; i++) s += data[i];
            mu = s / count;
            for (i = 0; i < count; i++) ss += (data[i] - mu)^2;
            std = (count > 1) ? sqrt(ss / (count - 1)) : 0;
            printf "%-40s: %.4f ± %.4f\n", name, mu, std;
          } else {
            printf "%-40s: N/A (No data)\n", name;
          }
        }

        END {
          print "==================================================================";
          printf " Final statistics for %s (noise=%s, %d runs)\n", dataset, noise, num_runs;
          print "==================================================================";
          print_stat("Perturbation Discrimination (PDS)", pds, c_pds);
          print_stat("Mean Absolute Error (MAE)",       mae, c_mae);
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

          # -------- 写 CSV：Method + 11(mean±std) + 每次 run 的 11 个原始值 --------
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
          for (i=1;i<=11;i++) { ms = mean_std(i); split(ms, parts, "|"); row = row sprintf(",%.4f±%.4f", parts[1], parts[2]); }
          for (r=0;r<num_runs;r++) for (i=1;i<=11;i++) row = row sprintf(",%.4f", val(i, r));

          print header > csv_path;
          print row    >> csv_path;
          close(csv_path);
          printf("CSV written: %s\n", csv_path);
        }
      '

      echo
      echo "--- Finished: cell=${cell_type} | noise=${noise_level} ---"
      echo
    } 2>&1 | tee -a "${log_file}"

  done
done

echo "######################################################################"
echo "###   All cells & noise-level pipelines are complete.               ###"
echo "######################################################################"
