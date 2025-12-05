<div align= "center">
    <h1> 🌊 PertDiffBench </h1>
</div>

## 📰 News
- Oct 2025 — Our paper “Benchmarking Diffusion Models for Predicting Perturbed Cellular Responses” has been accepted to the NeurIPS 2025 Workshop on Biosecurity Safeguards for Generative AI🎉🎉🎉!

## ⚙️ Configure the environment and prepare the data

### 🛠️ Configure the environment

```
conda create -n pertbench python=3.10 -y && conda activate pertbench
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install omegaconf numpy anndata tqdm scanpy gdown einops torch_geometric adjustText wandb 
pip install git+https://github.com/LouiseDck/scgen
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py
```

### 📥 Download the data and the pre-train model

#### 📊 Data

Data are still being organized...

## 📈 Evaluation

### Highly variable gene gradient

In the data of Task 1 in Figure 1, the CD4T cell type has the largest number of cells (5,564), and is therefore chosen as the representative.

First, run `python scripts/tools/get_the_hvg_data_for_fig1.py` to generate the hvg data. Then run

```
nohup bash scripts/highly_variable_gene_gradient/ddpm_hvg.sh > ddpm_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/ddpm_mlp_hvg.sh > ddpm_mlp_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scdiff_hvg.sh > scdiff_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scgen_hvg.sh > scgen_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/squidiff_hvg.sh > squidiff_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scdiffusion_hvg.sh > scdiffusion_hvg.log 2>&1
```

to obtain the evaluation results, respectively. The script will output the results from three experimental runs and their averaged results in the log, while also generating a CSV file for easy table completion.

### Fig 1

#### Task 1

**Get the data**

Since, overall, the models trained on the data with the lowest number of highly variable genes (1000) achieved the best performance, the experiments of Task 1 and Task 3 in Figure 1 are conducted using the processed data with 1000 HVGs extracted from the original data.  

First, run `python scripts/tools/get_the_hvg_data_for_fig3.py` to generate the data used in the Task 3 experiment of Figure 1. Then, organize this data together with the data obtained from the highly variable gene gradient experiments, for example:

```
/PertBench/
├── /data/
│  ├── /hvg_fig1/
│  │  └── B_train_HVG_1000.h5ad
│  ├── /hvg_fig3/
│  │  └── mix2_test_HVG_1000.h5ad
```

**Run the evaluation**

```bash
nohup bash scripts/fig1/fig1_task2_ddpm_mlp.sh > fig1_task2_ddpm_mlp.log 2>&1
nohup bash scripts/fig1/fig1_task2_ddpm.sh > fig1_task2_ddpm.log 2>&1
nohup bash scripts/fig1/fig1_task2_scgen.sh > fig1_task2_scgen.log 2>&1
nohup bash scripts/fig1/fig1_task2_scdiff.sh > fig1_task2_scdiff.log 2>&1
nohup bash scripts/fig1/fig1_task2_scdiffusion.sh > fig1_task2_scdiffusion.log 2>&1
nohup bash scripts/fig1/fig1_task2_squidff.sh > fig1_task2_squidff.log 2>&1
```

#### Task 2

**Run the evaluation**

```bash
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_ddpm_mlp.sh > fig1_task2_ddpm_mlp.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_ddpm.sh > fig1_task2_ddpm.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_scgen.sh > fig1_task2_scgen.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_scdiff.sh > fig1_task2_scdiff.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_scdiffusion.sh > fig1_task2_scdiffusion.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task2_squidff.sh > fig1_task2_squidff.log 2>&1
```

#### Task 3

**Run the evaluation**

```
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_ddpm_mlp.sh > fig1_task3_ddpm_mlp.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_ddpm.sh > fig1_task3_ddpm.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_scgen.sh > fig1_task3_scgen.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_scdiff.sh > fig1_task3_scdiff.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_scdiffusion.sh > fig1_task3_scdiffusion.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task3_squidff.sh > fig1_task3_squidff.log 2>&1
```

#### Task 4 

**Get the data**

1. 将 exp.csv 和 meta.csv 合并为 .h5ad 数据。运行

   ```bash
   bash scripts/tools/fig1_task4_merge.sh
   ```

   得到 `task4_ACTA2_control.h5ad`，`task4_ACTA2_coculture.h5ad`，`task4_ACTA2_IFN.h5ad`，`task4_B2M_control.h5ad`，`task4_B2M_coculture.h5ad`和`task4_B2M_IFN.h5ad`数据文件。

2. 划分方式 1：输入control预测coculture（训练集:测试集=8:2），输入control预测IFN（训练集:测试集=8:2）。运行

   ```bash
   bash scripts/tools/fig1_task4_split_1.sh
   ```

   得到`task4_B2M_control_coculture_train.h5ad`，`task4_B2M_control_coculture_test.h5ad`等共八个数据文件。注意，由于control和coculture（其他数据集也一样）的基因序列并不相同，直接合并会出现 nan 值，这里采用了取并集然后将 nan 变为 0 的通用做法。

3. 划分方式2：训练时control预测IFN，测试时control预测coculture。运行

   ```bash
   python scripts/tools/create_global_gene_list.py
   ```

   统一基因空间，基因数目为5737。然后运行

   ```bash
   bash scripts/tools/fig1_task4_split_2.sh
   ```

   得到`task4_ACTA2_control_to_coculture.h5ad`，`task4_ACTA2_control_to_ifn.h5ad`，`task4_B2M_control_to_coculture.h5ad`和`task4_B2M_control_to_ifn.h5ad`四个数据文件。

**1  Squidiff**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_squidiff.sh > fig1_task4_1_squidiff.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_squidiff.sh > fig1_task4_2_squidiff.log 2>&1
   ```

**2  scDiff**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scdiff.sh > fig1_task4_1_scdiff.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scdiff.sh > fig1_task4_2_scdiff.log 2>&1
   ```

**3  scDiffusion**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scdiffusion.sh > fig1_task4_1_scdiffusion.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scdiffusion.sh > fig1_task4_2_scdiffusion.log 2>&1
   ```

**4  scGen**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scgen.sh > fig1_task4_1_scgen.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scgen.sh > fig1_task4_2_scgen.log 2>&1
   ```

**5  DDPM**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_ddpm.sh > fig1_task4_1_ddpm.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_ddpm.sh > fig1_task4_2_ddpm.log 2>&1
   ```

**6  DDPM+MLP**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_ddpm_mlp.sh > fig1_task4_1_ddpm_mlp.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_ddpm_mlp.sh > fig1_task4_2_ddpm_mlp.log 2>&1
   ```

### Fig 2

#### Task 1

**0  获取数据**

将 exp.csv 和 meta.csv 合并为 .h5ad 数据，并合并为训练集和测试集。运行

```bash
bash scripts/tools/fig2_task1_merge.sh
```

得到`seed123_control_test.h5ad`、`seed123_control_train.h5ad`等数据集。

**1  Squidiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_squidiff.sh > fig2_task1_squidiff.log 2>&1
```

**2  scDiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scdiff.sh > fig2_task1_scdiff.log 2>&1
```

**3  scDiffusion**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scdiffusion.sh > fig2_task1_scdiffusion.log 2>&1
```

**3  scGen**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scgen.sh > fig2_task1_scgen.log 2>&1
```

**5  DDPM**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_ddpm.sh > fig2_task1_ddpm.log 2>&1
```

**6 DDPM+MLP**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_ddpm_mlp.sh > fig2_task1_ddpm_mlp.log 2>&1
```

#### Task 2

**1  Squidiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_squidiff.sh > fig2_task2_squidiff.log 2>&1
```

**2  scDiff**

受原代码限制，不进行该实验。

**3  scDiffusion**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_scdiffusion.sh > fig2_task2_scdiffusion.log 2>&1
```

**4  scGen**

受原代码限制，不进行该实验。

**5  DDPM**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_ddpm.sh > fig2_task2_ddpm.log 2>&1
```

**6 DDPM+MLP**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_ddpm_mlp.sh > fig2_task2_ddpm_mlp.log 2>&1
```

#### Task 3

**0  Get the data**

将 exp.csv 和 meta.csv 合并为 .h5ad 数据。运行

```bash
bash scripts/tools/fig2_task3.sh
```

You will get `mouse_control_ifn.h5ad`等四个数据。

**1  Squidiff**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_squidiff.sh > fig2_task3_squidiff.log 2>&1
```

**2  scDiff**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scdiff.sh > fig2_task3_scdiff.log 2>&1
```

**3  scDiffusion**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scdiffusion.sh > fig2_task3_scdiffusion.log 2>&1
```

**4  scGen**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scgen.sh > fig2_task3_scgen.log 2>&1
```

**5  DDPM**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_ddpm.sh > fig2_task3_ddpm.log 2>&1
```

**2  DDPM+MLP**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_ddpm_mlp.sh > fig2_task3_ddpm_mlp.log 2>&1
```

## 噪声扰动数据
### 高斯噪声扰动
运行
```
conda activate pertbench && export PYTHONPATH=./
cd scripts/tools/noise_perturbation_exp
python cd4t_gaus.py
```
你会得到高斯噪声扰动后的数据在 `data/add_gaussian_noise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm_mlp.sh > gausnoise_ddpm_mlp.log 2>&1
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm.sh > gausnoise_ddpm.log 2>&1
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiff.sh > gausnoise_scdiff.log 2>&1
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiffusion.sh > gausnoise_scdiffusion.log 2>&1
nohup bash scripts/noise_exp/gaussian_perturbed_data/scgen.sh > gausnoise_scgen.log 2>&1
nohup bash scripts/noise_exp/gaussian_perturbed_data/squidiff.sh > gausnoise_squidiff.log 2>&1
```

### 生物噪声（对数正态分布）
运行
```
conda activate pertbench && export PYTHONPATH=./
cd scripts/tools/noise_perturbation_exp
python cd4t_log_norm.py
```
你会得到生物噪声扰动后的数据在 `data/add_lognormal_bionoise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm_mlp.sh > lognormal_ddpm_mlp.log 2>&1
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm.sh > lognormal_ddpm.log 2>&1
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiff.sh > lognormal_scdiff.log 2>&1
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiffusion.sh > lognormal_scdiffusion.log 2>&1
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scgen.sh > lognormal_scgen.log 2>&1
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/squidiff.sh > lognormal_squidiff.log 2>&1
```

### 技术噪声
#### 泊松分布

运行
```
conda activate pertbench && export PYTHONPATH=./
cd scripts/tools/noise_perturbation_exp
python cd4t_poisson.py
```
你会得到技术噪声扰动后的数据在 `data/add_poisson_technoise_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm_mlp.sh > poisson_ddpm_mlp.log 2>&1
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm.sh > poisson_ddpm.log 2>&1
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiff.sh > poisson_scdiff.log 2>&1
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiffusion.sh > poisson_scdiffusion.log 2>&1
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scgen.sh > poisson_scgen.log 2>&1
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/squidiff.sh > poisson_squidiff.log 2>&1
```

#### 零膨胀模型
运行
```
conda activate pertbench && export PYTHONPATH=./
cd scripts/tools/noise_perturbation_exp
python cd4t_zero_inflation.py
```
你会得到技术噪声扰动后的数据在 `data/add_zero_inflation_output` 路径下。（你可能需要运行两次，以获得 train 数据和 valid 数据）

然后运行
```
cd ../../..
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm_mlp.sh > zero_inflation_ddpm_mlp.log 2>&1
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm.sh > zero_inflation_ddpm.log 2>&1
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiff.sh > zero_inflation_scdiff.log 2>&1
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiffusion.sh > zero_inflation_scdiffusion.log 2>&1
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scgen.sh > zero_inflation_scgen.log 2>&1
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/squidiff.sh > zero_inflation_squidiff.log 2>&1
```


## 编码器实验
### scVI
```
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scimilarity_ddpm.sh > encoder_scimilarity_ddpm.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1
conda activate pertbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scfoundation_ddpm.sh > encoder_scfoundation_ddpm.log 2>&1
conda activate scgpt && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scgpt_ddpm.sh > encoder_scgpt_ddpm.log 2>&1
conda activate geneformer && export PYTHONPATH=./
nohup bash scripts/encoder_exp/geneformer_ddpm.sh > encoder_geneformer_ddpm.log 2>&1
conda activate cellfm && export PYTHONPATH=./
nohup bash scripts/encoder_exp/cellfm/cellfm_ddpm.sh > encoder_cellfm_ddpm.log 2>&1
```
