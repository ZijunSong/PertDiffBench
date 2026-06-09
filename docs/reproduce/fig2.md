# Figure 2 Reproduction

### Fig 2

#### Task 1

**Data Preparation**

Merge `exp.csv` and `meta.csv` into `.h5ad` format and generate the corresponding training and test sets.

Run:

```bash
bash preprocess_data/fig2/task1_unseenPert/fig2_task1_merge.sh
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

**Task 2 — leave-one-out unseen cell type with partial test controls (Fig2 `task2_plus`)**

This extension uses all **seven** PBMC cell types from the Task 1 PBMC setup (`B`, `CD4T`, `CD8T`, `CD14+Mono`, `Dendritic`, `FCGR3A+Mono`, `NK`). For each **leave-one-out (LOO)** fold, one cell type is held out as the target; models are trained on the other six cell types (full Control + IFN) and, following scGen-style generalization, an additional **random fraction of the held-out type’s Control cells** is included in training. The **remaining** held-out Control cells, together with **all** IFN (stimulated) cells of that type, define the test split. Three training fractions are evaluated: **0%**, **25%**, and **50%** of the held-out type’s Control cells added to training.

**Data preparation**

Merge each cell type’s `task1_train_*` and `task1_valid_*` CSVs and emit LOO splits as `.h5ad` files:

```bash
python preprocess_data/fig2/task2_unseen_celltype_plus/task2_unseen_celltype_plus_loo.py \
  --ori-dir <path_to>/data_ori/fig2/task2_unseen_celltype_plus \
  --out-root <path_to>/data/fig2/task2_unseen_celltype_plus \
  --seed 0
```

For each `(held-out cell type, control fraction)` pair, the script writes `task2_train_exp.h5ad`, `task2_test_exp.h5ad`, and `scgen_combined_train_plus_test_control.h5ad` under `loo_<celltype>/<p0|p0.25|p0.5>/`. Point baselines at this tree via `DATA_BASE` (default `data/fig2/task2_unseen_celltype_plus` relative to the repo root).

**Evaluation grid:** Each baseline script loops over **7 LOO folds × 3 control fractions × `NUM_RUNS`** (default **`NUM_RUNS=3`**). That is **21** dataset configurations per method, each evaluated with **three** random repeats unless you change `NUM_RUNS`. Metrics are aggregated into one row per `(fold, fraction)` in the global CSV (mean ± std over runs).

**Note on scDiffusion:** For each of the **21** configurations, the VAE, diffusion, and classifier are trained **once**; `NUM_RUNS` then controls how many **sampling / evaluation** passes are averaged (other baselines typically perform a full train+eval per run index).

**scDiffusion pretrained VAE weights:** `fig2_task2_plus_scdiffusion.sh` passes `--state_dict` to the VAE trainer (encoder/decoder init). By default it uses `ANNOTATION_MODEL_DIR=/data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1` (expects `encoder.ckpt`, `decoder.ckpt`, `gene_order.tsv` there). To use a copy under the repo instead, run e.g. `export ANNOTATION_MODEL_DIR=/path/to/PertDiffBench/checkpoints/annotation_model_v1` before launching the script.

**Default GPUs:** The six scripts are configured for **single-GPU** runs on **GPU 0–5** by default (`scGen`→0, `scDiff`→1, DDPM→2, DDPM+MLP→3, Squidiff→4, scDiffusion→5). Override with `CUDA_VISIBLE_DEVICES` if needed.

**Run the evaluation** (from the repository root, after activating your environment and `export PYTHONPATH=./`):

```bash
cd /data/ppnm/PertDiffBench && conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_scgen.sh > fig2_task2_plus_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_scdiff.sh > fig2_task2_plus_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_ddpm.sh > fig2_task2_plus_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_ddpm_mlp.sh > fig2_task2_plus_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_squidiff.sh > fig2_task2_plus_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_scdiffusion.sh > fig2_task2_plus_scdiffusion.log 2>&1 &
```

**Squidiff (slow):** `fig2_task2_plus_squidiff.sh` runs the **p0** control fraction by default. For **p0.25** and **p0.5**, use the split wrappers (defaults: GPU 6 and 7):

```bash
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_squidiff_p0.25.sh > fig2_task2_plus_squidiff_p0.25.log 2>&1 &
nohup bash scripts/fig2/fig2_task2_plus/fig2_task2_plus_squidiff_p0.5.sh > fig2_task2_plus_squidiff_p0.5.log 2>&1 &
```

To run all three fractions in one process, set `FIG2_TASK2_PLUS_SQUIDIFF_SLUGS="p0 p0.25 p0.5"` before calling `fig2_task2_plus_squidiff.sh`.

#### Task 3

**Get the data**

Merge `exp.csv` and `meta.csv` into `.h5ad` files. Run:

```bash
bash preprocess_data/fig2/task3_cross_species/fig2_task3.sh
```

This produces four `.h5ad` datasets, including `mouse_control_ifn.h5ad`.

**Run the evaluation**

```bash
nohup bash scripts/fig2/fig2_task3/fig2_task3_squidiff.sh > fig2_task3_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scdiff.sh > fig2_task3_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scdiffusion.sh > fig2_task3_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_scgen.sh > fig2_task3_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_ddpm.sh > fig2_task3_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task3/fig2_task3_ddpm_mlp.sh > fig2_task3_ddpm_mlp.log 2>&1 &
```

In the journal extension, we additionally report results under a **leave-one-out** experimental setting:

```bash
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_squidiff.sh > fig2_task3_extend_squidiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scdiff.sh > fig2_task3_extend_scdiff.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scdiffusion.sh > fig2_task3_extend_scdiffusion.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_scgen.sh > fig2_task3_extend_scgen.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_ddpm.sh > fig2_task3_extend_ddpm.log 2>&1 &
nohup bash scripts/fig2/fig2_task3_extend/fig2_task3_extend_ddpm_mlp.sh > fig2_task3_extend_ddpm_mlp.log 2>&1 &
```
