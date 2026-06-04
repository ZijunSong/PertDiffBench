#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge paired *_exp.csv and *_meta.csv into .h5ad for two folders:
  - unseen_diff_moa/
  - unseen_same_moa/

For each pair, create an AnnData with:
  - X: expression matrix (cells x genes)
  - obs: metadata (+ perturbation_status="IFN")
  - var: gene names

Robust alignment:
  1) If there is a shared ID column between exp/meta, align by it (preferred).
  2) Else align by index and verify identical ordering.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import anndata as ad

# Raw and processed data roots (script can be run from repo root)
DATA_ORI = Path("/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA")
DATA_OUT = Path("/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA")

# Candidate columns that might represent a cell identifier
ID_CANDIDATES = [
    "cell_id", "cellid", "CellID", "CellId", "CELL_ID",
    "barcode", "Barcode", "BARCODE",
    "obs_id", "obsid", "ObsID",
]


def _detect_shared_id_col(df_exp: pd.DataFrame, df_meta: pd.DataFrame) -> Optional[str]:
    """Return a shared ID column name if both have it; else None."""
    exp_cols = set(df_exp.columns)
    meta_cols = set(df_meta.columns)
    for c in ID_CANDIDATES:
        if c in exp_cols and c in meta_cols:
            return c
    return None


def _load_exp(exp_path: Path) -> pd.DataFrame:
    """
    Load expression CSV.
    Tries to interpret first column as index; if that fails (e.g., cell_id is a column),
    falls back to normal read and leaves id as a column.
    """
    # First try: index_col=0 (common for expression matrices)
    df = pd.read_csv(exp_path, low_memory=False)
    # Heuristic: if first column looks like an ID column and remaining are many genes,
    # keep it as a normal column; otherwise treat first column as index.
    if df.shape[1] >= 2:
        first = df.columns[0]
        if first in ID_CANDIDATES:
            return df
    # If not obviously an ID column, try index_col=0
    df2 = pd.read_csv(exp_path, index_col=0, low_memory=False)
    # If index is purely numeric and a column named like gene exists, keep as df2 anyway;
    # numeric index can be legitimate, but usually cell IDs are strings.
    return df2


def _load_meta(meta_path: Path) -> pd.DataFrame:
    """
    Load metadata CSV similarly.
    """
    df = pd.read_csv(meta_path, low_memory=False)
    if df.shape[1] >= 1 and df.columns[0] in ID_CANDIDATES:
        return df
    # Common: index_col=0 as cell index
    df2 = pd.read_csv(meta_path, index_col=0, low_memory=False)
    return df2


def _align_exp_meta(
    df_exp: pd.DataFrame, df_meta: pd.DataFrame, exp_path: Path, meta_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align exp and meta by shared ID column if possible; else by index.
    Returns (exp_aligned, meta_aligned, gene_names)
    where exp_aligned is a DataFrame with index as cell ids and columns as genes.
    """
    warnings: List[str] = []

    # Case A: both have a shared ID column
    id_col = _detect_shared_id_col(df_exp, df_meta)
    if id_col is not None:
        # Determine gene columns in exp (everything except id_col)
        gene_cols = [c for c in df_exp.columns if c != id_col]
        if len(gene_cols) == 0:
            raise ValueError(f"No gene columns found in {exp_path} after excluding id col '{id_col}'.")

        exp_idx = df_exp[id_col].astype(str)
        meta_idx = df_meta[id_col].astype(str)

        df_exp2 = df_exp.copy()
        df_meta2 = df_meta.copy()

        df_exp2[id_col] = exp_idx
        df_meta2[id_col] = meta_idx

        # Set index
        df_exp2 = df_exp2.set_index(id_col)
        df_meta2 = df_meta2.set_index(id_col)

        # Intersect + order by exp
        common = df_exp2.index.intersection(df_meta2.index)
        if len(common) == 0:
            raise ValueError(
                f"Found shared id column '{id_col}' but no overlapping ids between:\n"
                f"  exp:  {exp_path}\n  meta: {meta_path}"
            )

        if len(common) < df_exp2.shape[0] or len(common) < df_meta2.shape[0]:
            warnings.append(
                f"[WARN] ID overlap not full for pair:\n"
                f"  exp rows={df_exp2.shape[0]} meta rows={df_meta2.shape[0]} common={len(common)}\n"
                f"  Using intersection (drops non-overlapping rows)."
            )

        df_exp_aligned = df_exp2.loc[common, gene_cols]
        df_meta_aligned = df_meta2.loc[common, :]

        return df_exp_aligned, df_meta_aligned, gene_cols

    # Case B: both already indexed
    if df_exp.index is None or df_meta.index is None:
        raise ValueError(
            f"Cannot align exp/meta for:\n  exp: {exp_path}\n  meta:{meta_path}\n"
            f"No shared ID column and missing index."
        )

    # Normalize to string indices (common in AnnData)
    df_exp_idx = df_exp.copy()
    df_meta_idx = df_meta.copy()
    df_exp_idx.index = df_exp_idx.index.astype(str)
    df_meta_idx.index = df_meta_idx.index.astype(str)

    # Prefer exact index match; otherwise use intersection and warn
    if df_exp_idx.index.equals(df_meta_idx.index):
        gene_cols = list(df_exp_idx.columns)
        return df_exp_idx, df_meta_idx, gene_cols

    common = df_exp_idx.index.intersection(df_meta_idx.index)
    if len(common) == 0:
        raise ValueError(
            f"Index mismatch and no overlap between:\n  exp:  {exp_path}\n  meta: {meta_path}"
        )

    warnings.append(
        f"[WARN] Index not identical for pair:\n"
        f"  exp rows={df_exp_idx.shape[0]} meta rows={df_meta_idx.shape[0]} common={len(common)}\n"
        f"  Using intersection (drops non-overlapping rows) and ordering by exp."
    )

    df_exp_aligned = df_exp_idx.loc[common, :]
    df_meta_aligned = df_meta_idx.loc[common, :]

    gene_cols = list(df_exp_aligned.columns)
    return df_exp_aligned, df_meta_aligned, gene_cols


def _make_out_name(exp_file: Path) -> str:
    """
    Convert 'Something_train_exp.csv' -> 'Something_train.h5ad'
    and 'Something_test_exp.csv' -> 'Something_test.h5ad'
    """
    name = exp_file.name
    name = re.sub(r"_exp\.csv$", "", name)
    return f"{name}.h5ad"


def process_folder(folder_in: Path, out_dir: Path, perturbation_status: str = "IFN") -> None:
    if not folder_in.exists():
        print(f"[SKIP] Folder does not exist: {folder_in}")
        return

    print(f"\n=== Processing folder: {folder_in} ===")

    exp_files = sorted(folder_in.glob("*_exp.csv"))
    if not exp_files:
        print(f"[WARN] No *_exp.csv files found in {folder_in}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_path in exp_files:
        meta_path = folder_in / Path(exp_path.name.replace("_exp.csv", "_meta.csv"))
        if not meta_path.exists():
            print(f"[WARN] Missing meta for exp: {exp_path.name} -> expected {meta_path.name}. Skipping.")
            continue

        out_path = out_dir / _make_out_name(exp_path)

        # Resume: skip if already exists
        if out_path.exists():
            print(f"[OK] Exists, skip: {out_path}")
            continue

        try:
            df_exp_raw = _load_exp(exp_path)
            df_meta_raw = _load_meta(meta_path)

            df_exp, df_meta, gene_cols = _align_exp_meta(df_exp_raw, df_meta_raw, exp_path, meta_path)

            # Ensure numeric expression
            X = df_exp.to_numpy()
            # Construct AnnData
            adata = ad.AnnData(
                X=X,
                obs=df_meta.copy(),
                var=pd.DataFrame(index=pd.Index(gene_cols, name="gene"))
            )

            adata.obs["perturbation_status"] = perturbation_status

            # A tiny bit of hygiene: store where it came from
            adata.uns["source_exp_csv"] = str(exp_path)
            adata.uns["source_meta_csv"] = str(meta_path)
            adata.uns["perturbation_status_value"] = perturbation_status

            adata.write(out_path)
            print(f"[OK] Wrote: {out_path}  (cells={adata.n_obs}, genes={adata.n_vars})")

        except Exception as e:
            print(f"[ERR] Failed on pair:\n  exp : {exp_path}\n  meta: {meta_path}\n  err : {e}")


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    diff_folder_in = DATA_ORI / "unseen_diff_moa"
    same_folder_in = DATA_ORI / "unseen_same_moa"
    diff_out = DATA_OUT / "unseen_diff_moa" / "h5ad"
    same_out = DATA_OUT / "unseen_same_moa" / "h5ad"

    process_folder(diff_folder_in, diff_out, perturbation_status="IFN")
    process_folder(same_folder_in, same_out, perturbation_status="IFN")

    print("\nDone.")


if __name__ == "__main__":
    main()
