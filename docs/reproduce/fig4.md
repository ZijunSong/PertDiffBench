# Figure 4 Reproduction

### Fig 4

#### Task 1 — Time-conditioned generation

**Data preparation**

Merge the Fig 4 expression matrix (CSV) with cell metadata into `.h5ad` files, then split by time point: **training** at 0h, 2h, 8h, and 10h; **test** at 4h and 6h.

From the repository root, run:

```bash
python preprocess_data/fig4/prepare_fig4_h5ad.py
```

By default the script reads `GSM3770930_A549_lognorm_scale_hvg3000.csv` and `GSM3770930_A549_cell_annotate.txt` from `data_ori/fig4/` (on this setup the absolute root is `/data/ppnm/data/PertDiffBench/data_ori/fig4/`) and writes:

- `/data/ppnm/data/PertDiffBench/data/fig4_task1/fig4_train.h5ad` — training cells, with `treatment_time` and a compatibility `perturbation_status` column
- `/data/ppnm/data/PertDiffBench/data/fig4_task1/fig4_test.h5ad` — held-out 4h/6h cells for comparing generated vs. real profiles at those time points

**Running evaluation**

```bash
conda activate pertdiffbench && export PYTHONPATH=./ && cd /path/to/PertDiffBench
nohup bash scripts/fig4/fig4_task1_scdiffusion.sh > fig4_task1_scdiffusion.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_squidiff.sh > fig4_task1_squidiff.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_ddpm.sh > fig4_task1_ddpm.log 2>&1 &
nohup bash scripts/fig4/fig4_task1_ddpm_mlp.sh > fig4_task1_ddpm_mlp.log 2>&1 &
```

Each launcher runs multiple training/evaluation rounds with time conditioning and writes per-method metrics under `samples/fig4/<method>/metrics_*_fig4*.csv`. To merge all baseline CSVs:

```bash
python scripts/fig4/aggregate_fig4_metrics.py
```

which produces `samples/fig4/fig4_metrics_merged.csv`. **scDiff** is not yet wired into the Fig 4 time-conditioned evaluation loop.

**Baseline settings (Fig 4 Task 1)**

- **scDiffusion** — Train VAE, diffusion, and classifier on the training set with `treatment_time` as the condition. At test time, synthesize 4h/6h cells via **classifier gradient interpolation** along the 2h→8h direction, then evaluate.
- **DDPM** — Train a DDPM in raw expression space on `fig4_train`, and train a **Fig 4–specific VAE** (encoder–decoder only, same architecture family as DDPM+MLP) using `train_fig4_ae_for_ddpm.py`. At test time, linearly interpolate 2h/8h in latent space, decode to 4h/6h, and compare to real 4h/6h cells.
- **DDPM+MLP** — Train encoder, latent diffusion, and decoder on the training set. At test time, use only this model’s encoder/decoder with **linear** 2h/8h interpolation to obtain 4h/6h (no diffusion sampling at inference).
- **Squidiff** — Train Squidiff on the training set. At test time, linearly interpolate 2h/8h in Squidiff latent space to get 4h/6h latents, then decode through the diffusion decoder to expression for evaluation.
