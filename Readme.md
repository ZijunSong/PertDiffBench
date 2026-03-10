<div align= "center">
    <h1> 🌊 PertDiffBench </h1>
</div>

## 📰 News
- Oct 2025 — Our paper “Benchmarking Diffusion Models for Predicting Perturbed Cellular Responses” has been accepted to the NeurIPS 2025 Workshop on Biosecurity Safeguards for Generative AI🎉🎉🎉!

## ⚙️ Configure the environment and prepare the data

### 🛠️ Configure the environment

```
conda create -n pertdiffbench python=3.10 -y && conda activate pertbench
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install omegaconf numpy anndata tqdm scanpy gdown einops torch_geometric adjustText wandb 
pip install git+https://github.com/LouiseDck/scgen
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py
```

### 📥 Download the data and pre-trained model

Pre-trained models for each baseline are available at [Google Drive (models)](https://drive.google.com/file/d/1ckeo3Ku0r1B9bk2yrzIgNOR3nyGpcGGE/view?usp=sharing), and the raw CSV data at [Google Drive (data)](https://drive.google.com/file/d/1-I6Je5nT5QcBm0AGn-mLUHKGx8_FZJdH/view?usp=sharing).

You can download them with the following commands:

```bash
pip install gdown

gdown "https://drive.google.com/uc?id=1ckeo3Ku0r1B9bk2yrzIgNOR3nyGpcGGE" --fuzzy
gdown "https://drive.google.com/uc?id=1-I6Je5nT5QcBm0AGn-mLUHKGx8_FZJdH" --fuzzy
```

## 📈 Evaluation

### Highly variable gene gradient

In the data of Task 1 in Figure 1, the CD4T cell type has the largest number of cells (5,564) and is therefore chosen as the representative.

**Preprocessing (in order):**

1. Run `python preprocess_data/fig1/fig1_task1.py` to convert the `.csv` files in `data_ori/fig1/raw_task1` into `.h5ad` format.
2. Run `python preprocess_data/get_the_hvg_data_for_fig1.py` to extract highly variable gene (HVG) data from the resulting `.h5ad` files.

**Evaluation:** Run the following commands to obtain results for each baseline:

```
nohup bash scripts/highly_variable_gene_gradient/ddpm_hvg.sh > ddpm_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/ddpm_mlp_hvg.sh > ddpm_mlp_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scdiff_hvg.sh > scdiff_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scgen_hvg.sh > scgen_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/squidiff_hvg.sh > squidiff_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scdiffusion_hvg.sh > scdiffusion_hvg.log 2>&1 &
```

Each script runs three trials and writes the per-run and averaged metrics to the log, and also outputs a CSV file for easy tabulation.

**Note:** You may need to set your own data and model paths (e.g. via the `ROOT_DIR` and `ANNOTATION_MODEL_DIR` environment variables or the defaults at the top of each script).

### Fig 1

#### Task 1

**Get the data**

Since, overall, the models trained on the data with the lowest number of highly variable genes (1000) achieved the best performance, the experiments of Task 1 and Task 3 in Figure 1 are conducted using the processed data with 1000 HVGs extracted from the original data.  

**Run the evaluation**

```bash
nohup bash scripts/fig1/fig1_task1/fig1_task1_ddpm_mlp.sh > fig1_task1_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_ddpm.sh > fig1_task1_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scgen.sh > fig1_task1_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scdiff.sh > fig1_task1_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scdiffusion.sh > fig1_task1_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_squidff.sh > fig1_task1_squidff.log 2>&1 &
```

#### Task 2

**Run the evaluation**

```bash
nohup bash scripts/fig1/fig1_task2_ddpm_mlp.sh > fig1_task2_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_ddpm.sh > fig1_task2_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scgen.sh > fig1_task2_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scdiff.sh > fig1_task2_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scdiffusion.sh > fig1_task2_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_squidff.sh > fig1_task2_squidff.log 2>&1 &
```

#### Task 3

**Run the evaluation**

```
nohup bash scripts/fig1/fig1_task3_ddpm_mlp.sh > fig1_task3_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_ddpm.sh > fig1_task3_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scgen.sh > fig1_task3_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scdiff.sh > fig1_task3_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scdiffusion.sh > fig1_task3_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_squidff.sh > fig1_task3_squidff.log 2>&1 &
```

#### Task 4 

**Get the data**

1. **Merge `exp.csv` and `meta.csv` into `.h5ad` format**

   Run the following script:

   ```bash
   bash preprocess_data/fig1_task4_merge.sh
   ```

   This will generate the following `.h5ad` files: `task4_ACTA2_control.h5ad`, `task4_ACTA2_coculture.h5ad`, `task4_ACTA2_IFN.h5ad`, `task4_B2M_control.h5ad`, `task4_B2M_coculture.h5ad`, `task4_B2M_IFN.h5ad`.

2. **Split Strategy 1**

   * Train on *control* and predict *coculture* (train:test = 8:2)
   * Train on *control* and predict *IFN* (train:test = 8:2)

   Run:

   ```bash
   bash preprocess_data/fig1_task4_split_1.sh
   ```

   This will generate eight `.h5ad` files, including: `task4_B2M_control_coculture_train.h5ad`, `task4_B2M_control_coculture_test.h5ad` (and corresponding files for other gene groups).

   **Note:**
   Since the gene sets in *control* and *coculture* (and similarly in other dataset pairs) are not identical, directly merging them would introduce `NaN` values. To address this, we take the union of genes and replace missing values (`NaN`) with zeros as a standard preprocessing step.

3. **Split Strategy 2**

   * Training: control → IFN
   * Testing: control → coculture

   First, unify the gene space:

   ```bash
   python preprocess_data/create_global_gene_list.py
   ```

   This produces a unified gene list containing **5,737 genes**.

   Then run:

   ```bash
   bash preprocess_data/fig1_task4_split_2.sh
   ```

   This will generate four `.h5ad` files: `task4_ACTA2_control_to_coculture.h5ad`, `task4_ACTA2_control_to_ifn.h5ad`, `task4_B2M_control_to_coculture.h5ad`, `task4_B2M_control_to_ifn.h5ad`.

**Run the evaluation**
```
nohup bash scripts/fig1/fig1_task4_1_squidiff.sh > fig1_task4_1_squidiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_squidiff.sh > fig1_task4_2_squidiff.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scdiff.sh > fig1_task4_1_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scdiff.sh > fig1_task4_2_scdiff.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scdiffusion.sh > fig1_task4_1_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scdiffusion.sh > fig1_task4_2_scdiffusion.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scgen.sh > fig1_task4_1_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scgen.sh > fig1_task4_2_scgen.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_ddpm.sh > fig1_task4_1_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_ddpm.sh > fig1_task4_2_ddpm.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_ddpm_mlp.sh > fig1_task4_1_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_ddpm_mlp.sh > fig1_task4_2_ddpm_mlp.log 2>&1 &
```

### Fig 2

#### Task 1

**Data Preparation**

Merge `exp.csv` and `meta.csv` into `.h5ad` format and generate the corresponding training and test sets.

Run:

```bash
bash preprocess_data/fig2_task1_merge.sh
```

This will produce datasets such as: `seed123_control_train.h5ad`, `seed123_control_test.h5ad` (and other datasets generated with the same random seed).

**Run the evaluation**

```
nohup bash scripts/fig2/fig2_task1/fig2_task1_squidiff.sh > fig2_task1_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1/fig2_task1_scdiff.sh > fig2_task1_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1/fig2_task1_scdiffusion.sh > fig2_task1_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task1/fig2_task1_scgen.sh > fig2_task1_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task1/fig2_task1_ddpm.sh > fig2_task1_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task1/fig2_task1_ddpm_mlp.sh > fig2_task1_ddpm_mlp.log 2>&1 &
```

**MOA Classification Evaluation (Journal Extension)**

In the journal extension, we evaluate MOA classification using the dataset located at `data/fig2/task1_unseenMOA`. The evaluation contains two experimental settings.

(1) Same-MOA Split

Under the **Same-MOA split**, each MOA (Mechanism of Action) is split at the drug level into predefined training and testing subsets. Each MOA contains: `<MOA>_train_exp.csv`, `<MOA>_train_meta.csv`, `<MOA>_test_exp.csv`, `<MOA>_test_meta.csv`.

For this setting, the model is trained and evaluated independently for each MOA. If there are 15 MOAs, the training–evaluation procedure is executed 15 times.

(2) Diff-MOA Split 

Under the **Diff-MOA split**, one MOA is held out for testing while the remaining MOAs are used for training.

For example:

* Train on MOAs A–E
* Test on MOA F

The process is repeated in a leave-one-out manner over all MOAs. The file format remains identical: `<MOA>_train_exp.csv`, `<MOA>_train_meta.csv`, `<MOA>_test_exp.csv`, `<MOA>_test_meta.csv`.

To facilitate downstream modeling, we provide a unified preprocessing pipeline that converts all CSV files into standardized `.h5ad` format and constructs Control+IFN merged datasets.

```bash
python preprocess_data/fig2/task1_unseenMOA/control_csv_to_h5ad.py
python preprocess_data/fig2/task1_unseenMOA/merge_unseen_moa_to_h5ad.py
python preprocess_data/fig2/task1_unseenMOA/merge_control_with_each_ifn.py
python preprocess_data/fig2/task1_unseenMOA/add_smiles_from_chembl.py
python preprocess_data/fig2/task1_unseenMOA/check_pairs.py
```

The pipeline performs the following steps in order:

1. Convert the control CSV files into `control_merged.h5ad` and write it to the output directory.
2. Convert IFN-only data for each MOA into `.h5ad` format, writing to:
   * `unseen_same_moa/h5ad/`
   * `unseen_diff_moa/h5ad/`
3. Merge each MOA with the control dataset, producing:
   * `control_plus_ifn/<split>/<MOA>_train__plus_control.h5ad`
   * `control_plus_ifn/<split>/<MOA>_test__plus_control.h5ad`
4. Add SMILES from ChEMBL to the h5ad files in `control_plus_ifn`, writing results to `control_plus_ifn_with_smiles/`.
5. Verify dataset integrity (e.g. `perturbation_status`, Control/IFN sample counts, and merged sample sizes).

After preprocessing, all resulting `.h5ad` files are ready for training and evaluation under both Same-MOA and Unseen-MOA settings.

**Path convention for training scripts:** Input data is read from `DATA_BASE` (default `/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA`). Checkpoints are written to `/data/ppnm/checkpoints/PertDiffBench/checkpoints/fig2/task1_unseenMOA/<same|diff>/<method>/...`. Generated samples and metrics are written to `/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/<same|diff>/<method>/<dataset>/run{i}`. You can override `DATA_BASE`, `SAMPLES_BASE`, or `CKPT_BASE` via environment variables if needed.

Run from the project root (e.g. after `conda activate pertdiffbench && export PYTHONPATH=./`):

```bash
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_squidiff_moa_same.sh > fig2_task1_squidiff_moa_same.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_scdiff_moa_same.sh > fig2_task1_scdiff_moa_same.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_scdiffusion_moa_same.sh > fig2_task1_scdiffusion_moa_same.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_ddpm_moa_same.sh > fig2_task1_ddpm_moa_same.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_ddpm_mlp_moa_same.sh > fig2_task1_ddpm_mlp_moa_same.log 2>&1 &

nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_squidiff_moa_diff.sh > fig2_task1_squidiff_moa_diff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_scdiff_moa_diff.sh > fig2_task1_scdiff_moa_diff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_scdiffusion_moa_diff.sh > fig2_task1_scdiffusion_moa_diff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_ddpm_moa_diff.sh > fig2_task1_ddpm_moa_diff.log 2>&1 &
nohup bash scripts/fig2/fig2_task1_moa/fig2_task1_ddpm_mlp_moa_diff.sh > fig2_task1_ddpm_mlp_moa_diff.log 2>&1 &
```

#### Task 2

In this task, we train the model on one cell type and evaluate it on different cell types, aiming to assess cross-cell-type generalization.
**Experiment setting:** models are trained on **CD4 T cells** and evaluated on **B** and **NK** cells (unseen cell types).

We first conduct a **fully out-of-distribution (OOD) generalization setting**, where no data from the test cell type is used during training.
Since `scDiff` and `scGen` require pre-perturbation data from the test domain during training for their OOD experiments, they are excluded under this strictly OOD protocol.

The following scripts reproduce this fully OOD setting:

```bash
nohup bash scripts/fig2/fig2_task2/fig2_task2_squidiff.sh > fig2_task2_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task2/fig2_task2_scdiffusion.sh > fig2_task2_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task2/fig2_task2_ddpm.sh > fig2_task2_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task2/fig2_task2_ddpm_mlp.sh > fig2_task2_ddpm_mlp.log 2>&1 &
```

We then conduct a second experimental setting in which **pre-perturbation data from both the training and test cell types are included during training**.
Under this partially shared setting, all methods, including `scDiff` and `scGen`, are evaluated.

The corresponding scripts are:

```bash
cd /share/PertBench && conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_squidiff.sh > fig2_task2_extend_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_scdiffusion.sh > fig2_task2_extend_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_ddpm.sh > fig2_task2_extend_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_ddpm_mlp.sh > fig2_task2_extend_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_scdiff.sh > fig2_task2_extend_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_extend/fig2_task2_extend_scgen.sh > fig2_task2_extend_scgen.log 2>&1 &
```

#### Task 3

**0  Get the data**

将 exp.csv 和 meta.csv 合并为 .h5ad 数据。运行

```bash
bash preprocess_data/fig2_task3.sh
```

You will get `mouse_control_ifn.h5ad`等四个数据。

**Run the Evaluation**

```bash
nohup bash scripts/fig2/fig2_task3/fig2_task3_squidiff.sh > fig2_task3_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scdiff.sh > fig2_task3_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scdiffusion.sh > fig2_task3_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scgen.sh > fig2_task3_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_ddpm.sh > fig2_task3_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_ddpm_mlp.sh > fig2_task3_ddpm_mlp.log 2>&1 &
```

在期刊版本，我们又新增了留一法实验设置的结果。

```bash
cd /share/PertBench &&conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_squidiff.sh > fig2_task3_extend_squidiff.log 2>&1 &

cd /share/PertBench && conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scdiff.sh > fig2_task3_extend_scdiff.log 2>&1 &

nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scdiffusion.sh > fig2_task3_extend_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scgen.sh > fig2_task3_extend_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_ddpm.sh > fig2_task3_extend_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_ddpm_mlp.sh > fig2_task3_extend_ddpm_mlp.log 2>&1 &
```

### Fig 4

#### Task 1（时间条件生成）

**Data Preparation**

将 fig4 的表达矩阵 CSV 与元数据合并为 `.h5ad`，并按时间点划分训练集（0h, 2h, 8h, 10h）与测试集（4h, 6h）。

Run:

```bash
python data/fig4/prepare_fig4_h5ad.py
```

默认会读取同目录下的 `GSM3770930_A549_lognorm_scale_hvg3000.csv` 与 `GSM3770930_A549_cell_annotate.txt`，并生成：

- `data/fig4/fig4_train.h5ad`（训练集，含 `treatment_time` 与兼容用 `perturbation_status`）
- `data/fig4/fig4_test.h5ad`（测试集，用于与生成结果 4h/6h 比较）

**Run the evaluation**

```bash
conda activate pertbench && export PYTHONPATH=./ && cd /share/PertBench
nohup bash scripts/fig4/fig4_task1_scdiffusion.sh > fig4_task1_scdiffusion.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_squidiff.sh > fig4_task1_squidiff.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_ddpm.sh > fig4_task1_ddpm.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_ddpm_mlp.sh > fig4_task1_ddpm_mlp.log 2>&1 &
```

各脚本会对 fig4 数据做多轮训练与时间条件评估，并将汇总结果写入 `samples/fig4/<method>/metrics_*_fig4*.csv`。合并所有 baseline 的 CSV：`python scripts/fig4/aggregate_fig4_metrics.py`，输出为 `samples/fig4/fig4_metrics_merged.csv`。scDiff 当前未接入 fig4 时间条件评估。

**Fig4 各 baseline 设定说明**

- **scDiffusion**：训练集训练 VAE + diffusion + classifier（`treatment_time` 为条件）。测试集 4h/6h 由 classifier **梯度插值**（2h–8h 方向）生成后评估。
- **DDPM**：训练集用 fig4_train 训练原始表达空间的 DDPM；并为 fig4 **单独训练一个 VAE**（仅 encoder+decoder，与 DDPM+MLP 同结构，脚本 `train_fig4_ae_for_ddpm.py`）。测试集：用该 VAE 做 2h/8h latent 线性插值 → decoder 得到 4h/6h，再与真实 4h/6h 评估。
- **DDPM+MLP**：训练集训练 encoder + 潜空间扩散 + decoder。测试集仅用本模型 encoder/decoder 做 2h/8h 线性插值生成 4h/6h，不跑 diffusion。
- **Squidiff**：训练集训练 Squidiff；测试集在 Squidiff latent 空间对 2h/8h 线性插值得到 4h/6h latent，再经 diffusion 解码为表达后评估。

## 噪声扰动数据
### 高斯噪声扰动
运行
```
conda activate pertbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_gaus.py
```
你会得到高斯噪声扰动后的数据在 `data/add_gaussian_noise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm_mlp.sh > gausnoise_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm.sh > gausnoise_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiff.sh > gausnoise_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiffusion.sh > gausnoise_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scgen.sh > gausnoise_scgen.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/squidiff.sh > gausnoise_squidiff.log 2>&1 &
```

### 生物噪声（对数正态分布）
运行
```
conda activate pertbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_log_norm.py
```
你会得到生物噪声扰动后的数据在 `data/add_lognormal_bionoise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm_mlp.sh > lognormal_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm.sh > lognormal_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiff.sh > lognormal_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiffusion.sh > lognormal_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scgen.sh > lognormal_scgen.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/squidiff.sh > lognormal_squidiff.log 2>&1 &
```

### 技术噪声
#### 泊松分布

运行
```
conda activate pertbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_poisson.py
```
你会得到技术噪声扰动后的数据在 `data/add_poisson_technoise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm_mlp.sh > poisson_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm.sh > poisson_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiff.sh > poisson_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiffusion.sh > poisson_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scgen.sh > poisson_scgen.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/squidiff.sh > poisson_squidiff.log 2>&1 &
```

#### 零膨胀模型
运行
```
conda activate pertbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_zero_inflation.py
```
你会得到技术噪声扰动后的数据在 `data/add_zero_inflation_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm_mlp.sh > zero_inflation_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm.sh > zero_inflation_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiff.sh > zero_inflation_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiffusion.sh > zero_inflation_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scgen.sh > zero_inflation_scgen.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/squidiff.sh > zero_inflation_squidiff.log 2>&1 &
```


## 编码器实验
```bash
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1 &
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scimilarity_ddpm.sh > encoder_scimilarity_ddpm.log 2>&1 &
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1 &
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scfoundation_ddpm.sh > encoder_scfoundation_ddpm.log 2>&1 &
conda activate scgpt && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scgpt_ddpm.sh > encoder_scgpt_ddpm.log 2>&1 &
conda activate geneformer && export PYTHONPATH=./
nohup bash scripts/encoder_exp/geneformer_ddpm.sh > encoder_geneformer_ddpm.log 2>&1 &
```

### CellFM

MindSpore 2.6 GPU 需要 **CUDA 11**（libcublas.so.11）与 **cuDNN 8**。请用 **conda-forge** 的 `cudatoolkit`（勿用 nvidia 的 `cuda-toolkit`，易被解析成 CUDA 13）：

```bash
# 推荐：新建环境并用 conda-forge 装 CUDA 11 + cuDNN 8
conda create -n cellfm_cuda11 python=3.10 -y
conda activate cellfm_cuda11
conda install -y -c conda-forge cudatoolkit=11.7.1 cudnn=8.4.1.50
pip install mindspore==2.6.0 -i https://pypi.org/simple
pip install huggingface_hub scanpy anndata
# 检查（应有输出）
ls $CONDA_PREFIX/lib/libcublas.so.11* $CONDA_PREFIX/lib/libcudnn.so.8* 2>/dev/null
```

或仅用 CPU 环境（不装 cudatoolkit/cudnn）时无法跑 CellFM 编码，需换带 GPU 的机器。

```bash
conda activate cellfm_cuda11 && export PYTHONPATH=./   # 或 cellfm
nohup bash scripts/encoder_exp/cellfm/cellfm_ddpm.sh > encoder_cellfm_ddpm.log 2>&1 &
```

### tahoe-x1

```bash
conda create -n tahoe-x1 python=3.10 -y && conda activate tahoe-x1

pip install torch scanpy omegaconf anndata "numpy<2"
pip install mosaicml-streaming   
pip install composer              
pip install boto3 transformers datasets  
pip install llm-foundry
```

```bash
cd /share/PertBench && conda activate tahoe-x1 && export PYTHONPATH=./
nohup bash scripts/encoder_exp/tahoe-x1/tx1_ddpm.sh > encoder_tx1_ddpm.log 2>&1 &
```

### State

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install arc-state
```

```bash
cd /share/PertBench && conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/state/state_ddpm.sh > encoder_state_ddpm.log 2>&1 &
```
