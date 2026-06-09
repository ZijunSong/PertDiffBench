<div align="center">
    <h1>🌊 PertDiffBench</h1>
    <p><b>Benchmarking Diffusion Models for Predicting Perturbed Cellular Responses</b></p>
</div>

## News

- **Oct 2025** — Accepted to the NeurIPS 2025 Workshop on Biosecurity Safeguards for Generative AI.

## Overview

PertDiffBench evaluates diffusion and baseline models on perturbation-response prediction. The **`pertdiffbench`** Python package provides a unified interface to run multiple models across benchmark tasks with one command.

## Supported Models

| Registry name | Method |
|---------------|--------|
| `ddpm` | Expression-space conditional DDPM |
| `ddpm_mlp` | Latent DDPM + MLP autoencoder |
| `squidiff` | Squidiff semantic latent diffusion |
| `scdiffusion` | scDiffusion (VAE + classifier-guided diffusion) |
| `scgen` | scGen VAE perturbation model |
| `scdiff` | scDiff diffusion model |
| `chemcpa` | ChemCPA (MOA tasks only) |
| `encoder` | Pretrained encoder + latent DDPM (encoder task only) |

## Supported Tasks

| Task | Description |
|------|-------------|
| `known_condition` | Control → IFN, in-distribution |
| `cross_celltype` | Strict OOD: train CD4T, test unseen types |
| `cross_celltype_extend` | scGen-style partial test controls in training |
| `cross_celltype_plus` | LOO cell type + control fraction (p0/p0.25/p0.5) |
| `cross_species` | Train mouse, test other species |
| `cross_species_loo` | Leave-one-species-out |
| `moa_same` | Same-MOA unseen drug |
| `moa_diff` | Cross-MOA unseen mechanism |
| `temporal` | A549 time-point imputation (Fig 4) |
| `noise` | Robustness under synthetic noise |
| `encoder` | External encoder + latent DDPM |

## Quick Start

```bash
conda activate pertdiffbench
cd PertDiffBench
pip install -e .

# List tasks and compatible models
pertdiffbench list-tasks
pertdiffbench list-models-for-task --task known_condition

# Run multiple models on known-condition prediction
pertdiffbench run \
  --task known_condition \
  --train data/highly_variable_gene_gradient/CD4T_train_HVG_1000.h5ad \
  --test  data/highly_variable_gene_gradient/CD4T_valid_HVG_1000.h5ad \
  --models ddpm,ddpm_mlp,squidiff,scdiffusion,scgen,scdiff \
  --output runs/cd4t_demo \
  --gene-nums 1000 --n-samples 278 --num-runs 1
```

### MOA example

```bash
pertdiffbench run \
  --task moa_same \
  --data-root data/fig2_task1_unseenMOA/control_plus_ifn_with_smiles/unseen_same_moa \
  --models ddpm,ddpm_mlp,squidiff,chemcpa \
  --use-drug-structure \
  --output runs/moa_same --gene-nums 3000
```

### Encoder example

```bash
pertdiffbench run \
  --task encoder \
  --encoder scgpt \
  --train data/fig1/raw_task1/task1_train_CD4T_exp.h5ad \
  --test  data/fig1/raw_task1/task1_valid_CD4T_exp.h5ad \
  --encoder-ckpt-dir /path/to/scgpt/checkpoint \
  --output runs/encoder_scgpt
```

### Python API

```python
from pertdiffbench import BenchmarkRunner

runner = BenchmarkRunner(task="known_condition", output_dir="runs/exp")
runner.run(
    models=["ddpm", "squidiff", "scgen"],
    train_h5ad="data/.../train.h5ad",
    test_h5ad="data/.../test.h5ad",
    gene_nums=1000,
    num_runs=3,
)
```

## Documentation

- [Installation](docs/installation.md)
- [Data download](docs/data.md)
- [Full reproduction guide](docs/index.md)

## Citation

If you use PertDiffBench, please cite our paper *(bibtex to be added)*.
