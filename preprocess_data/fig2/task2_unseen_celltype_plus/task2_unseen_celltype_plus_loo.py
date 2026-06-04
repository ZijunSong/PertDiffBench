#!/usr/bin/env python3
"""
Fig2 task2+: preprocess PBMC data for leave-one-out cross-cell-type perturbation prediction.

- Merge task1_train_* / task1_valid_* CSV per cell type (cells x genes).
- Each fold: hold out one cell type; training = all cells from the other 6 types plus a fraction p of
  held-out Control cells; testing = remaining (1-p) held-out Control cells plus all IFN (stimulated).
- Writes task2_train_exp.h5ad, task2_test_exp.h5ad, scgen_combined_train_plus_test_control.h5ad per fold.

Example:
  python task2_unseen_celltype_plus_loo.py \\
    --ori-dir /path/to/data_ori/fig2/task2_unseen_celltype_plus \\
    --out-root /path/to/PertDiffBench/data/fig2/task2_unseen_celltype_plus \\
    --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# Seven PBMC cell types (same convention as fig1_task1)
CELL_TYPES: List[str] = [
    "B",
    "CD4T",
    "CD8T",
    "CD14+Mono",
    "Dendritic",
    "FCGR3A+Mono",
    "NK",
]

CTRL_FRACS_DEFAULT = (0.0, 0.25, 0.5)


def _slug_frac(p: float) -> str:
    if abs(p - 0.0) < 1e-9:
        return "p0"
    if abs(p - 0.25) < 1e-9:
        return "p0.25"
    if abs(p - 0.5) < 1e-9:
        return "p0.5"
    s = f"{p:g}".replace(".", "_")
    return f"p{s}"


def _read_exp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def _csv_to_adata(exp_df: pd.DataFrame, cell_type: str) -> ad.AnnData:
    """Build AnnData from a cells x genes matrix (same as fig1_task1)."""
    adata = ad.AnnData(exp_df, dtype=np.float32)
    adata.obs["Cell.Type"] = cell_type
    idx = adata.obs_names.astype(str)
    is_ctrl = idx.str.endswith("-control")
    is_stim = idx.str.endswith("-stimulated")
    adata.obs["perturbation_status"] = np.where(
        is_stim, "IFN", np.where(is_ctrl, "Control", "unknown")
    )
    return adata


def _merge_train_valid(ori_dir: str, cell_type: str) -> ad.AnnData:
    train_p = os.path.join(ori_dir, f"task1_train_{cell_type}_exp.csv")
    valid_p = os.path.join(ori_dir, f"task1_valid_{cell_type}_exp.csv")
    if not os.path.isfile(train_p):
        raise FileNotFoundError(train_p)
    if not os.path.isfile(valid_p):
        raise FileNotFoundError(valid_p)
    tdf = _read_exp_csv(train_p)
    vdf = _read_exp_csv(valid_p)
    overlap = tdf.index.intersection(vdf.index)
    if len(overlap):
        # Official splits may still list the same barcodes in train and valid; keep train rows.
        print(
            f"  [info] {cell_type}: {len(overlap)} cell IDs appear in both train and valid; "
            f"keeping train rows only."
        )
        vdf = vdf.drop(index=overlap, errors="ignore")
    merged = pd.concat([tdf, vdf], axis=0)
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    return _csv_to_adata(merged, cell_type)


def _intersect_genes(adatas: Sequence[ad.AnnData]) -> List[str]:
    genes = set(adatas[0].var_names.astype(str))
    for a in adatas[1:]:
        genes &= set(a.var_names.astype(str))
    if not genes:
        raise RuntimeError("Empty gene intersection across cell types.")
    return sorted(genes)


def _subset_genes(adata: ad.AnnData, genes: List[str]) -> ad.AnnData:
    g = [g for g in genes if g in adata.var_names]
    return adata[:, g].copy()


def _build_combined_for_ddpm(train_adata: ad.AnnData, test_adata: ad.AnnData) -> ad.AnnData:
    """Combined h5ad for DDPM: all train cells split=train; append test Control as split=test_control (fig2_task2_extend)."""
    tr = train_adata.copy()
    tr.obs["split"] = "train"
    te_ctrl = test_adata[test_adata.obs["perturbation_status"].astype(str) == "Control"].copy()
    te_ctrl.obs["split"] = "test_control"
    comb = sc.concat([tr, te_ctrl], join="inner", index_unique=None)
    comb.obs_names_make_unique()
    return comb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ori-dir",
        type=str,
        default="/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus",
        help="Directory with raw CSVs (task1_train_* / task1_valid_* per cell type)",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Output root (creates loo_<CT>/<p*>/). Default: <repo>/data/fig2/task2_unseen_celltype_plus",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fracs",
        type=float,
        nargs="*",
        default=list(CTRL_FRACS_DEFAULT),
        help="Fractions of held-out Control cells added to training (e.g. 0 0.25 0.5)",
    )
    args = p.parse_args()

    ori_dir = os.path.abspath(args.ori_dir)
    if not args.out_root:
        here = os.path.dirname(os.path.abspath(__file__))
        proj = os.path.normpath(os.path.join(here, "..", "..", ".."))
        out_root = os.path.join(proj, "data", "fig2", "task2_unseen_celltype_plus")
    else:
        out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading & merging per cell type from {ori_dir} ...")
    by_ct: Dict[str, ad.AnnData] = {}
    for ct in CELL_TYPES:
        by_ct[ct] = _merge_train_valid(ori_dir, ct)
        print(f"  {ct}: {by_ct[ct].n_obs} cells × {by_ct[ct].n_vars} genes")

    common_genes = _intersect_genes(list(by_ct.values()))
    print(f"Gene intersection size: {len(common_genes)}")
    for ct in CELL_TYPES:
        by_ct[ct] = _subset_genes(by_ct[ct], common_genes)

    rng = np.random.default_rng(args.seed)

    for holdout in CELL_TYPES:
        others = [c for c in CELL_TYPES if c != holdout]
        held = by_ct[holdout]
        ctrl_mask = held.obs["perturbation_status"].astype(str) == "Control"
        stim_mask = held.obs["perturbation_status"].astype(str) == "IFN"
        ctrl_idx = np.where(ctrl_mask)[0]
        n_ctrl = int(ctrl_idx.size)
        if n_ctrl < 2:
            print(f"[skip] {holdout}: not enough control cells ({n_ctrl})", file=sys.stderr)
            continue
        stim_adata = held[stim_mask].copy()

        for frac in args.fracs:
            if frac < 0 or frac > 1:
                raise ValueError(f"Invalid frac {frac}")
            slug = _slug_frac(float(frac))
            subdir = os.path.join(out_root, f"loo_{holdout}", slug)
            os.makedirs(subdir, exist_ok=True)

            n_to_train = int(np.floor(n_ctrl * float(frac)))
            shuffled = rng.permutation(ctrl_idx)
            train_ctrl_idx = np.sort(shuffled[:n_to_train])
            test_ctrl_idx = np.sort(shuffled[n_to_train:])

            if test_ctrl_idx.size == 0:
                print(
                    f"[skip] {holdout} {slug}: no control left for test (frac={frac})",
                    file=sys.stderr,
                )
                continue

            parts_tr: List[ad.AnnData] = [by_ct[c].copy() for c in others]
            if n_to_train > 0:
                parts_tr.append(held[train_ctrl_idx].copy())

            train_adata = sc.concat(parts_tr, join="inner", index_unique=None)
            train_adata.obs_names_make_unique()

            parts_te = [held[test_ctrl_idx].copy(), stim_adata.copy()]
            test_adata = sc.concat(parts_te, join="inner", index_unique=None)
            test_adata.obs_names_make_unique()

            comb = _build_combined_for_ddpm(train_adata, test_adata)

            tp = os.path.join(subdir, "task2_train_exp.h5ad")
            ep = os.path.join(subdir, "task2_test_exp.h5ad")
            cp = os.path.join(subdir, "scgen_combined_train_plus_test_control.h5ad")
            train_adata.write_h5ad(tp, compression="gzip")
            test_adata.write_h5ad(ep, compression="gzip")
            comb.write_h5ad(cp, compression="gzip")

            print(
                f"Written {subdir}: train={train_adata.n_obs}, test={test_adata.n_obs}, "
                f"combined={comb.n_obs} (held-out={holdout}, ctrl_train_frac={frac})"
            )

    print(f"Done. Output under: {out_root}")


if __name__ == "__main__":
    main()
