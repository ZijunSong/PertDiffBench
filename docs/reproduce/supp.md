# Supplementary Experiments

## Supplementary experiments (`supp/`)

This section documents the **gene column ordering** supplementary analysis on **CD4T**, **Fig 1 Task 1** (known-condition Control → IFN prediction with **1000 HVGs**). The goal is to test whether benchmark scores change when gene columns are permuted while keeping the same genes and expression values.

| Condition | Description |
|-----------|-------------|
| **Natural HVG rank** (optional baseline) | Original column order from the HVG-gradient pipeline (`data/highly_variable_gene_gradient/`). Not rewritten by the supp preprocessor; use as a reference when comparing to main Fig 1 Task 1. |
| **`shuffle`** | Random permutation of gene columns (fixed seed). Train and valid share the same order. |
| **`cluster`** | Columns reordered by hierarchical clustering on train-set gene–gene Pearson correlation (average linkage). Train and valid share the same order. |

Scripts live under `supp/`; run artifacts go to `data/gene_order_exp/`, `checkpoints/.../supp/`, and `samples/supp/` (all ignored by `.gitignore` except small helper scripts and merged metric CSVs in `supp/`).

### Prerequisites

1. **Fig 1 CD4T HVG data** from the highly-variable-gene gradient pipeline:
   - `data/highly_variable_gene_gradient/CD4T_train_HVG_1000.h5ad`
   - `data/highly_variable_gene_gradient/CD4T_valid_HVG_1000.h5ad` (or align from full valid split; see preprocessor).
2. **Validation split** for gene alignment (recommended):
   - `data/fig1_task1/task1_valid_CD4T_exp.h5ad`

Generate the HVG files first if needed (see [Highly variable gene gradient](#highly-variable-gene-gradient) above).

### Data preparation

From the repository root:

```bash
conda activate pertdiffbench && export PYTHONPATH=./
cd /path/to/PertDiffBench

python supp/preprocess_reorder_genes_cd4t.py --mode both
```

Defaults:

- **Input train HVG**: `data/highly_variable_gene_gradient/`
- **Input valid (for alignment)**: `data/fig1_task1/task1_valid_CD4T_exp.h5ad`
- **Output**: `data/gene_order_exp/shuffle/` and `data/gene_order_exp/cluster/`  
  Each contains `CD4T_train_HVG_1000.h5ad`, `CD4T_valid_HVG_1000.h5ad`, and `CD4T_gene_order_<mode>.json`.

Useful flags:

```bash
python supp/preprocess_reorder_genes_cd4t.py --mode shuffle   # shuffle only
python supp/preprocess_reorder_genes_cd4t.py --mode cluster  # cluster only
python supp/preprocess_reorder_genes_cd4t.py --seed 42 --src-dir <path> --out-root <path> --valid-src <path>
```

(Optional) To include the **natural HVG rank** in downstream metric merging, symlink or copy the original HVG pair into `data/gene_order_exp/hvg_rank/` with the same filenames as above.

### Run evaluation

Each baseline runs **`NUM_RUNS=3`** (default) train+eval loops on the reordered train/valid pair. Override paths if needed:

- `ROOT_DIR` — data root (default `/data/ppnm/data/PertDiffBench/`)
- `CKPT_ROOT` — checkpoint root (default `/data/ppnm/checkpoints/PertDiffBench/checkpoints`)
- `NUM_RUNS` — number of repeated runs per method

**GPU layout** (one model per GPU; run **shuffle** batch first, then **cluster**):

| GPU | Method |
|-----|--------|
| 0 | DDPM+MLP |
| 1 | DDPM |
| 2 | scGen |
| 3 | scDiff |
| 4 | Squidiff |
| 5 | scDiffusion |

```bash
cd /path/to/PertDiffBench
mkdir -p supp/logs/shuffle supp/logs/cluster

# Batch A — shuffle (6-way parallel)
nohup bash supp/shuffle/ddpm_mlp.sh    > supp/logs/shuffle/ddpm_mlp.log    2>&1 &
nohup bash supp/shuffle/ddpm.sh        > supp/logs/shuffle/ddpm.log        2>&1 &
nohup bash supp/shuffle/scgen.sh       > supp/logs/shuffle/scgen.log       2>&1 &
nohup bash supp/shuffle/scdiff.sh      > supp/logs/shuffle/scdiff.log      2>&1 &
nohup bash supp/shuffle/squidiff.sh    > supp/logs/shuffle/squidiff.log    2>&1 &
nohup bash supp/shuffle/scdiffusion.sh > supp/logs/shuffle/scdiffusion.log 2>&1 &

# Wait for Batch A to finish, then Batch B — cluster (same GPU mapping)
nohup bash supp/cluster/ddpm_mlp.sh    > supp/logs/cluster/ddpm_mlp.log    2>&1 &
nohup bash supp/cluster/ddpm.sh        > supp/logs/cluster/ddpm.log        2>&1 &
nohup bash supp/cluster/scgen.sh       > supp/logs/cluster/scgen.log       2>&1 &
nohup bash supp/cluster/scdiff.sh      > supp/logs/cluster/scdiff.log      2>&1 &
nohup bash supp/cluster/squidiff.sh    > supp/logs/cluster/squidiff.log    2>&1 &
nohup bash supp/cluster/scdiffusion.sh > supp/logs/cluster/scdiffusion.log 2>&1 &
```

Per-method launchers mirror the main benchmark: `supp/shuffle/*.sh` and `supp/cluster/*.sh` call the same `scripts/baseline_exp/` trainers and evaluators as Fig 1 Task 1, with `GENE_ORDER` set via `supp/common/lib.sh`.

A printable copy of the nohup commands is in `supp/run_all_nohup.sh`.

**Outputs (under `ROOT_DIR`, default layout):**

| Path | Content |
|------|---------|
| `checkpoints/supp/<shuffle\|cluster>/...` | Per-run model weights |
| `samples/supp/<shuffle\|cluster>/CD4T/<method>/...` | Synthetic h5ad, UMAP plots, per-method `metrics_*.csv` |
| `supp/logs/<shuffle\|cluster>/` | Training/eval logs (gitignored) |

### Merge metrics

After all runs finish, aggregate per-method CSVs into one table:

```bash
python supp/merge_gene_order_metrics.py
```

This writes:

- `supp/gene_order_cd4t_metrics_merged.csv` — full metrics plus per-run columns
- `supp/gene_order_cd4t_metrics_summary.csv` — key metrics only (`gene_order` × `Method`)

To merge additional orders (e.g. natural HVG rank):

```bash
python supp/merge_gene_order_metrics.py --orders hvg_rank shuffle cluster
```

### `supp/` directory layout

```
supp/
├── preprocess_reorder_genes_cd4t.py   # build shuffle/cluster h5ad
├── merge_gene_order_metrics.py        # merge sample CSVs
├── run_all_nohup.sh                   # reference launch commands
├── common/lib.sh                      # shared paths, conda, metric aggregation
├── common/aggregate_metrics.awk
├── shuffle/                           # per-baseline scripts (shuffled gene order)
├── cluster/                           # per-baseline scripts (cluster gene order)
├── gene_order_cd4t_metrics_merged.csv # example merged results (if committed)
└── gene_order_cd4t_metrics_summary.csv
```
