# Encoder Experiments

## Encoder experiments

From the repository root (adjust conda env names to match your install):

```bash
conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1 &
conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scimilarity_ddpm.sh > encoder_scimilarity_ddpm.log 2>&1 &
conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scvi_ddpm.sh > encoder_scvi_ddpm.log 2>&1 &
conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scfoundation_ddpm.sh > encoder_scfoundation_ddpm.log 2>&1 &
conda activate scgpt && export PYTHONPATH=./
nohup bash scripts/encoder_exp/scgpt_ddpm.sh > encoder_scgpt_ddpm.log 2>&1 &
conda activate geneformer && export PYTHONPATH=./
nohup bash scripts/encoder_exp/geneformer_ddpm.sh > encoder_geneformer_ddpm.log 2>&1 &
```

### CellFM

MindSpore 2.6 on GPU expects **CUDA 11** (`libcublas.so.11`) and **cuDNN 8**. Install CUDA with **`cudatoolkit` from conda-forge**; avoid NVIDIA’s `cuda-toolkit` metapackage here, which may pull in CUDA 13 and break the stack.

```bash
# Recommended: fresh env with conda-forge CUDA 11 + cuDNN 8
conda create -n cellfm_cuda11 python=3.10 -y
conda activate cellfm_cuda11
conda install -y -c conda-forge cudatoolkit=11.7.1 cudnn=8.4.1.50
pip install mindspore==2.6.0 -i https://pypi.org/simple
pip install huggingface_hub scanpy anndata
# Sanity check (should list matching libraries)
ls $CONDA_PREFIX/lib/libcublas.so.11* $CONDA_PREFIX/lib/libcudnn.so.8* 2>/dev/null
```

A CPU-only setup (no `cudatoolkit` / cuDNN) cannot run CellFM encoding on GPU; use a GPU machine or a properly configured CUDA 11 environment.

```bash
conda activate cellfm_cuda11 && export PYTHONPATH=./   # or your `cellfm` env
nohup bash scripts/encoder_exp/cellfm/cellfm_ddpm.sh > encoder_cellfm_ddpm.log 2>&1 &
```

### Tahoe-X1

```bash
conda create -n tahoe-x1 python=3.10 -y && conda activate tahoe-x1

pip install torch scanpy omegaconf anndata "numpy<2"
pip install mosaicml-streaming
pip install composer
pip install boto3 transformers datasets
pip install llm-foundry
```

```bash
cd /path/to/PertDiffBench && conda activate tahoe-x1 && export PYTHONPATH=./
nohup bash scripts/encoder_exp/tahoe-x1/tx1_ddpm.sh > encoder_tx1_ddpm.log 2>&1 &
```

### State

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install arc-state
```

```bash
cd /path/to/PertDiffBench && conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/encoder_exp/state/state_ddpm.sh > encoder_state_ddpm.log 2>&1 &
```
