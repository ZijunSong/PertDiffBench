# Fig4 time-conditioned generation — data processing

## 1. Raw data

Directory: `data_ori/fig4/` (local mirror may be under `/data/ppnm/data/PertDiffBench/data_ori/fig4/`)

- **Expression matrix**: `GSM3770930_A549_lognorm_scale_hvg3000.csv` (cells × genes, log2-normalized, 3000 HVGs)
- **Metadata**: `GSM3770930_A549_cell_annotate.txt` with columns such as `sample`, `treatment_time`, `doublet_score`
- **Time points**: 0h, 2h, 4h, 6h, 8h, 10h (~1k–1.4k cells per time point)

## 2. `perturbation_status` (compatibility only)

- This task has **no** true Control/IFN groups—only `treatment_time`.
- To satisfy code that still reads `perturbation_status` (e.g. scDiffusion classifier), use this **train-only convention**:
  - **Train**: `0h, 2h` → `perturbation_status = "Control"`; `8h, 10h` → `perturbation_status = "IFN"`.
- Early time points stand in for Control and late for IFN **only for API compatibility**; **time-conditioned training/eval uses `treatment_time`** (0h/2h/8h/10h as multi-class labels).
- **Test** (4h, 6h) is not used for training; set `perturbation_status` to `"IFN"` or `"Holdout"` as a placeholder. Evaluate by `treatment_time` (4h vs 6h).

**Summary**

- Labeling 0h/2h as Control and 8h/10h as IFN on the train set is reasonable for compatibility.
- The test set does not need real Control/IFN—keep `treatment_time` for evaluation.

## 3. Train / test split

- **Train**
  - Cells from **0h, 2h, 8h, 10h** only.
  - Train time-conditioned models (e.g. scDiffusion diffusion + classifier on time).

- **Test**
  - Cells from **4h, 6h** only.
  - Evaluate generation at 4h and 6h vs ground truth (MMD, LISI, etc.).

You do **not** need a validation split from 0h/2h/8h/10h unless you want early stopping; optionally hold out ~10% of train time points as `valid`.

## 4. Recommended h5ad outputs

Output directory: `data/fig4_task1/` (from `preprocess_data/fig4/prepare_fig4_h5ad.py`)

| File | Contents | Use |
|------|----------|-----|
| `fig4_train.h5ad` | 0h + 2h + 8h + 10h cells | Training (VAE + diffusion + classifier, etc.) |
| `fig4_test.h5ad` | 4h + 6h cells | Evaluate vs real 4h/6h |

Optional: `fig4_valid.h5ad` (~10% held out from train time points).

## 5. Recommended `obs` columns

- **Required**
  - `treatment_time`: strings `"0h"`, `"2h"`, `"4h"`, `"6h"`, `"8h"`, `"10h"` for training labels and eval grouping.

- **Pipeline compatibility**
  - `perturbation_status`:
    - Train: 0h/2h → `"Control"`, 8h/10h → `"IFN"`.
    - Test: all `"IFN"` or `"Holdout"`.

- **Optional**
  - `split`: `"train"` / `"test"` (or `"valid"`).
  - Other metadata (e.g. `doublet_score`) as needed.

## 6. Difference from fig1 (brief)

- **fig1**: real Control/IFN per cell; train/valid by cell type or random split; paired Control/IFN common.
- **fig4**: time only; **split by time point**—train times (0h,2h,8h,10h) vs held-out (4h,6h); `perturbation_status` filled on train only for compatibility.

## 7. Scripts and evaluation (`scripts/fig4`)

- **Preprocessing**: `preprocess_data/fig4/prepare_fig4_h5ad.py` builds `fig4_train.h5ad` and `fig4_test.h5ad` from `data_ori/fig4`.
- **Time-conditioned eval**: `scripts/fig4/eval_fig4_time_conditioned.py` computes 11 metrics per `treatment_time` (same as fig1); requires `--test-h5ad`, `--generated-h5ad`; optional `--train-h5ad` (0h as control for delta metrics).
- **Baseline shells**: `fig4_task1_scdiffusion.sh`, `fig4_task1_ddpm.sh`, `fig4_task1_ddpm_mlp.sh`, `fig4_task1_scdiff.sh`, `fig4_task1_squidiff.sh`—multi-run train/eval and CSV aggregation like fig1.
- **Test without Control/IFN**: `fig4_test` is 4h/6h only. If a baseline eval requires both Control and IFN in test, use **time-conditioned mode**: `--time-conditioned` + `--generated-h5ad` and `eval_fig4_time_conditioned.py`.

## 8. How each method generates 4h/6h (aligned with fig2)

| Method | 4h/6h generation |
|--------|------------------|
| **DDPM** | Train a fig4-only VAE (encoder+decoder, same structure as DDPM+MLP): `train_fig4_ae_for_ddpm.py`; linear interp 2h/8h → 4h/6h: `sample_fig4_vae_linear_interp.py`. |
| **DDPM+MLP** | Built-in encoder/decoder, 2h/8h linear interp → 4h/6h (no diffusion at sample time); `sample_fig4_vae_linear_interp.py`, ckpt `model_epoch_1000.pth`. |
| **Squidiff** | **Addition**: origin at 2h, Δz_sem = mean(z_8h)−mean(z_2h), scale 1/3 (4h) and 2/3 (6h), `interp_with_direction` → z_mod, DDIM decode. Script: `sample_fig4_squidiff_interp.py` (`--method addition`, optional `lerp`). |
| **scDiffusion** | Classifier **gradient interpolation** along 2h–8h (not linear latent interp). |
