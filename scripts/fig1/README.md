# Figure 1 Scripts

Paper reproduction for Figure 1 (known-condition prediction, bulk RNA-seq, etc.).

**Full instructions:** [docs/reproduce/fig1.md](../../docs/reproduce/fig1.md)

Launchers live in subdirectories:

- `fig1_task1/` — PBMC cell-type benchmark (7 cell types)
- `fig1_task4/` — ACTA2 / B2M coculture experiments

From the repo root:

```bash
conda activate pertdiffbench && export PYTHONPATH=./
nohup bash scripts/fig1/fig1_task1/fig1_task1_ddpm.sh > fig1_task1_ddpm.log 2>&1 &
```
