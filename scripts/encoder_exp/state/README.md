# State + DDPM Encoder Experiment

Use the **State Embedding (SE)** model from [State](https://github.com/ArcInstitute/state) as the encoder with DDPM for perturbation prediction:

```
pre-perturbation scRNA -> State SE encoder -> DDPM -> post-perturbation scRNA
```

## Environment setup

### 1. Install State

```bash
uv tool install arc-state
```

Or from source:

```bash
git clone https://github.com/ArcInstitute/state.git
cd state
uv tool install -e .
```

### 2. Download SE-600M pretrained weights

Download from [HuggingFace](https://huggingface.co/arcinstitute/SE-600M), e.g.:

```bash
huggingface-cli download arcInstitute/SE-600M --local-dir /path/to/SE-600M
```

Expected layout:

- `se600m_epoch15.ckpt` or another `.ckpt`
- Model config files

## Run

From the PertBench repo root:

```bash
cd /share/PertBench

# Optional model paths (default: checkpoints/SE-600M)
export STATE_MODEL_DIR=/path/to/SE-600M
export STATE_CHECKPOINT=/path/to/SE-600M/se600m_epoch15.ckpt  # optional; auto-picks .ckpt if unset

bash scripts/encoder_exp/state/state_ddpm.sh
```

Or edit `STATE_MODEL_DIR` and `STATE_CHECKPOINT` in `state_ddpm.sh`.

## Outputs

- Encoded latents: `samples/encoder_exp/state_ddpm/task1_train_CD4T_with_state_latent.h5ad`, etc.
- DDPM checkpoints: `checkpoints/state_ddpm/latent_ddpm/run_1/`, etc.
- Metrics: `samples/encoder_exp/state_ddpm/metrics_CD4T.csv`
