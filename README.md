<div align= "center">
    <h1> 🌊 Perturbation Modeling with Diffusion Models Benchmark </h1>
</div>

Welcome to our new work: Perturbation Modeling with Diffusion Models Benchmark, the paper has been uploaded to the arxiv.


## File Structure
After downloading the data, the directory structure should look like this:
```
/PertBench/
├── /diffusion_baselines/
│  ├── /checkpoints/
│  ├── /configs/
│  │  └── ddpm_default.yaml
│  ├── /datasets/
│  │  ├── /CIFAR10/
│  │  ├── /scrna_data/
│  │  │  ├── NK_IFN_exp.csv
│  │  │  ├── NK_IFN_meta.csv
│  │  │  └── scrna.h5ad
│  │  ├── cifar10.py
│  │  └── scrna.py
│  ├── /logs/
│  ├── /models/
│  │  ├── base.py
│  │  ├── ddpm_model.py
│  │  ├── ddpm.py
│  │  ├── gaussian_diffusion.py
│  │  └── latent_diffusion.py
│  ├── /samples/
│  ├── /schedulers/
│  │  └── warmup.py
│  ├── /scripts/
│  │  ├── csv_to_h5ad.py
│  │  └── train_ddpm.py
│  ├── /trainers/
│  │  ├── bae_trainer.py
│  │  ├── ddpm_trainer.py
│  │  └── scrna_trainer.py
├── /scDiff/
├── /scDiffusion/
...
```
## ⚙️ Configure the environment and prepare the data
### 📥 Download the data
### 🛠️ Configure the environment
```
conda create -n pertbench python=3.10 -y && conda activate pertbench
pip install omegaconf numpy anndata tqdm scanpy
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
#### CellOT
- The original paper：
- Repo：
You can run the following code to easily configure the environment
```shell
cd CellOt
conda create -n cellot python=3.9.5 && conda activate cellot

conda update -n base -c defaults conda
pip install --upgrade pip
pip install -r requirements.txt
python setup.py develop
```
To train the CellOT model, you should run
```shell
python ./scripts/train.py \
  --outdir ./results/PRJNA/drug-cisplatin/model-cellot \
  --config ./configs/tasks/PRJNA.yaml \
  --config ./configs/models/cellot.yaml \
  --config.data.target cisplatin
```
Once trained, the model can be evaluated via:
```shell
python ./scripts/evaluate.py \
  --outdir ./results/4i/drug-cisplatin/model-cellot \
  --setting iid \
  --where data_space
```
#### scDiffusion
- The original paper：scDiffusion: conditional generation of high-quality single-cell data using diffusion model
- Repo：https://github.com/EperLuo/scDiffusion
1. Configure the environment
```shell
cd scDiffusion
conda create -n scdiffusion python=3.10 -y && conda activate scdiffusion
pip install torch==1.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html 
pip install -r requirements.txt
conda install mpi4py==3.1.4
pip install blobfile
```
The data used for training the model is formatted in h5ad. You can download the dataset that used in the original paper in https://figshare.com/s/49b29cb24b27ec8b6d72. If you want to add your own data, you need to add your data import code `cell_datasets_YOURDATA.py` to the `guided_diffusion` folder.
2. Train
To train the Autoencoder, please run:
```shell
cd VAE 
python VAE_train.py \
  --data_dir '/share/PurpNight/scDiffusion/guided_diffusion/data/scdiffusion_benchmark3_part_1.h5ad' \
  --num_genes 3702 \
  --save_dir '../output/checkpoint/VAE/benchmark' \
--max_steps 200000
```
Set the parameters *data_dir* and *save_dir* to your local path, and set the *num_genes* parameter to match the gene number of your dataset. Set the *state_dict* to the path where you store your downloaded scimilarity checkpoint. You can also train the autoencoder from scratch, this might need larger interation steps (larger than 1.5e5 steps would be good).

To train the diffusion backbone, please run:
```
python cell_train.py \
  --data_dir '/share/PurpNight/scDiffusion/guided_diffusion/data/scdiffusion_benchmark3_part_1.h5ad' \
  --vae_path 'output/checkpoint/VAE/benchmark/model_seed=0_step=199999.pt' \
  --model_name 'scDiffusionBenchmark' \
  --save_dir 'output/checkpoint/backbone' \
  --lr_anneal_steps 600000
```
First, set the parameters *vae_path* to the path of your trained Autoencoder. Next, set the *data_dir*, *model_name*(the folder to save the ckpt), and *save_dir*(the path to place the *model_name* folder). We trained the backbone for 6e5 steps.

To train the classifier, please run 
```
python classifier_train.py \
  --data_dir '/share/PurpNight/scDiffusion/guided_diffusion/data/scdiffusion_benchmark3_part_1.h5ad' \
  --model_path "output/checkpoint/classifier/benchmark_classifier" \
  --iterations 200000 \
  --vae_path 'output/checkpoint/VAE/benchmark/model_seed=0_step=199999.pt' \
  --num_class=13
```
Set the parameters *vae_path* to the path of your trained Autoencoder. Set the *num_class* parameter to match the number of classes in your dataset. Then, set the *model_path* to the path you would like to save the ckpt and execute the file. We trained the classifier for 2e5 steps.

3. Generate new samples

Unconditional generation:
```
python cell_sample.py \
  --model_path 'output/checkpoint/backbone/scDiffusionBenchmark/model600000.pt' \
  --sample_dir 'output/simulated_samples/benchmark'\
  --num_samples 3000 \
  --batch_size 1000
```
set the *model_path* to match the trained backbone model's path and set the *sample_dir* to your local path. The *num_samples* is the number of cell to generate, and the *batch_size* is the number of cell generate in one diffusion reverse process.

Running the file will generate new latent embeddings for the scRNA-seq data and save them in a .npz file. You can decode these latent embeddings and retrieve the complete gene expression data using `exp_script/script_diffusion_umap.ipynb` or `exp_script/script_static_eval.ipynb`.

Conditional generation:

Run `classifier_sample.py`: set the *model_path* and *classifier_path* to match the trained backbone model and the trained classifier, respectively. Also, set the *sample_dir* to your local path. The condition can be set in "main" (the param *cell_type* in the main() function refer to the cell_type you want to generate.). Running the file will generate new latent embeddings under the given conditions.

For example: `python classifier_sample.py --model_path 'output/checkpoint/backbone/my_diffusion/model600000.pt' --classifier_path 'output/checkpoint/classifier/my_classifier/model200000.pt' --sample_dir 'output/simulated_samples/muris' --num_samples 3000 --batch_size 1000`

You can decode these embeddings the same way as in unconditional generation.

For multi-conditional generation and gradiante interpolation, refer to the comments in the main() function and create_argparser() function (see the comments with *** mark).

**Experiments reproduce:**

The scripts in the exp_script/ directory can be used to reproduce the results presented in the paper. You can refer the process in any of these scripts to rebuild the gene expression from latent space. The `exp_script/down_stream_analysis_muris.ipynb` can reproduce the marker genes result. The `exp_script/script_diffusion_umap_multi-condi.ipynb` can reproduce the result of two-conditonal generation. The `exp_script/script_diffusion_umap_trajectory.ipynb` can reproduce the result of Gradient Interpolation. The `exp_script/script_diffusion_umap.ipynb` can reproduce the UMAP shown in the paper. The `exp_script/script_static_eval.ipynb` can reproduce the statistical metrics mentioned in the paper.

#### scFoundation
- The original paper：Large-scale foundation model on single-cell transcriptomics
- Repo：https://github.com/biomap-research/scFoundation

1. Configure the environment

```
git clone https://github.com/biomap-research/scFoundation.git
cd scFoundation

conda create -n scfoundation python=3.10 -y
conda activate scfoundation

pip install argparse numpy pandas scipy einops scanpy local_attention
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

2. 转换基因符号（🧐如果使用的是演示数据，跳过此步骤）

如果使用自己的基因表达数据，需要将基因符号转换为与模型要求的基因列表一致。可以使用`get_embedding.py`中的`main_gene_selection`函数来完成这一任务。

```
import pandas as pd
from get_embedding import main_gene_selection

# 加载基因列表
gene_list_df = pd.read_csv('../OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

# 假设 X_df 是你的单细胞数据（行是细胞，列是基因）
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)

# 保存数据
X_df.to_csv('your_data.csv', index=False)  # 或者保存为 .npy 格式
```

3. 推理

通过链接 https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA 下载模型权重文件，并将其放入 `models` 文件夹。

通过链接 https://doi.org/10.6084/m9.figshare.24049200 下载原始的基因表达数据示例，解压后命名为 `examples`。

## scELMo

1. 配置环境

```
git clone https://github.com/HelloWorldLTY/scELMo.git
cd scELMo

conda create -n scelmo python=3.8 -y
conda activate scelmo

pip install openai
pip install scib scib_metrics==0.3.3 mygene scanpy==1.9.3 scikit-learn

apt-get install -y python-setuptools python-pip #may not need it for HPC base
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
pip install .
```

pip install numpy==1.24.4

pip install jax==0.3.25

## scGen

1. 配置环境

```
git clone https://github.com/theislab/scgen.git
cd scgen

conda create -n scgen python=3.9 -y
conda activate scgen

#conda deactivate
#conda remove -n scgen --all -y

#pip install torch --index-url https://download.pytorch.org/whl/cu121 
#pip install -e .[dev,docs]

git clone https://github.com/theislab/scgen-reproducibility.git
cd scgen-reproducibility/code
pip install wget
python DataDownloader.py # 下载数据
```

2. 训练scGen

```
pip install keras==2.3.1
pip install tensorflow==1.15
pip install typing-extensions # Re
pip install get-version==2.2
pip install anndata
pip install scanpy
pip install protobuf==3.20
pip install adjustText

python ModelTrainer.py all
python ModelTrainer.py PCA
python ModelTrainer.py VecArithm
python ModelTrainer.py CVAE
python ModelTrainer.py scGen # 运行python ./train_scGen.py
python ModelTrainer.py STGAN # 运行python ./st_gan.py train
```

3. 复现图 2

```
pip install numpy
pip install pandas
pip install anndata
pip install scanpy
pip install scgen
pip install requests
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip --version
pip install pip==24.0
pip --version
pip install scvi-tools==0.17
pip install scipy

export PYTHONPATH=/share/PurpNight/scgen/scgen-reproducibility
python Fig2.py
```

## scGPT

### 1. Configure the environment

```
git clone https://github.com/bowang-lab/scGPT.git && cd scGPT
conda create -n scgpt python=3.9 -y && conda activate scgpt

conda deactivate && conda remove -n scgpt --all -y

pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install scgpt
pip install wandb
pip install numpy==1.25.2
pip install anndata==0.10.8
pip install ipython
```

### 2. Download the pretrained model

The original author recommends using the `whole-human` model by default in most applications, so only the [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) to the `whole-human` model is provided here. If you need other models, please refer to the original code repository. You should put the 3 files under the `examples/save/scGPT_bc/` folder.

### 3. Fine-tune

```
cd examples
export CUDA_VISIBLE_DEVICES=0,1,2,3
python finetune_integration.py
```

You may meet several errors, for example: 

```
File "/opt/mamba/envs/scgpt/lib/python3.9/site-packages/scvi/data/_built_in_data/_pbmc.py", line 81, in _load_pbmc_dataset    barcodes_metadata = pbmc_metadata["barcodes"].index.values.ravel().astype(np.str)
```

You can find the original file and change this part of the code:

```
vim /opt/mamba/envs/scgpt/lib/python3.9/site-packages/scvi/data/_built_in_data/_pbmc.py
```

On line 81 of the file, replace `np.str` with `str`. On line 89 of the file, replace `np.bool` with `bool`.
### 📥 Download the data
You can download the xxx dataset to the data path by
```shell
pip install gdown
```
