#!/usr/bin/env python3
"""
Fig 4 time-conditioned generation -- data prep (setup 1, scDiffusion-style).

Merge expression CSV + metadata into h5ad and split by time point:
- fig4_train.h5ad: 0h, 2h, 8h, 10h -- train models conditioned on time
- fig4_test.h5ad: 4h, 6h -- evaluation only (compare generated vs real 4h/6h)

perturbation_status (for downstream compatibility):
- Train: 0h/2h -> Control, 8h/10h -> IFN
- Test: 4h/6h -> IFN placeholder; evaluation uses treatment_time
"""

import argparse
import os
import sys

import pandas as pd
import anndata as ad


# Same layout as fig1 preprocess: raw files under data_ori, h5ad under a dedicated dir
DATA_ORI = "/data/ppnm/data/PertDiffBench/data_ori/fig4"
DATA_OUT = "/data/ppnm/data/PertDiffBench/data/fig4_task1"
DEFAULT_EXP = os.path.join(DATA_ORI, "GSM3770930_A549_lognorm_scale_hvg3000.csv")
DEFAULT_META = os.path.join(DATA_ORI, "GSM3770930_A549_cell_annotate.txt")
TRAIN_TIMES = {"0h", "2h", "8h", "10h"}
TEST_TIMES = {"4h", "6h"}


def _norm_time(s: str) -> str:
    """Normalize time strings: strip whitespace."""
    if pd.isna(s):
        return s
    return str(s).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Fig 4: merge expression + metadata, split by time, write train/test h5ad"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=DEFAULT_EXP,
        help="Expression matrix CSV (cells x genes), row index = cell ID",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=DEFAULT_META,
        help="Metadata table CSV with sample and treatment_time",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DATA_OUT,
        help=f"Output directory (default: {DATA_OUT})",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Write h5ad without gzip compression",
    )
    args = parser.parse_args()

    # 1) Load expression
    if not os.path.isfile(args.exp):
        print(f"Error: expression file not found: {args.exp}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading expression: {args.exp}")
    exp_df = pd.read_csv(args.exp, index_col=0)
    exp_df.index = exp_df.index.astype(str).str.strip()
    print(f"  shape: {exp_df.shape} (cells x genes)")

    # 2) Load metadata (index = cell ID)
    if not os.path.isfile(args.meta):
        print(f"Error: metadata file not found: {args.meta}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading metadata: {args.meta}")
    meta_df = pd.read_csv(args.meta, index_col=0)
    meta_df.index = meta_df.index.astype(str).str.strip()
    if "treatment_time" not in meta_df.columns:
        print("Error: metadata missing column treatment_time", file=sys.stderr)
        sys.exit(1)
    meta_df["treatment_time"] = meta_df["treatment_time"].apply(_norm_time)
    print(f"  treatment_time counts:\n{meta_df['treatment_time'].value_counts().sort_index()}")

    # 3) Align cells
    common = exp_df.index.intersection(meta_df.index)
    if len(common) == 0:
        print("Error: no overlapping cell IDs between expression and metadata", file=sys.stderr)
        sys.exit(1)
    exp_df = exp_df.loc[common]
    meta_df = meta_df.loc[common]
    print(f"Cells after alignment: {len(common)}")

    # 4) Time points
    times = set(meta_df["treatment_time"].dropna().unique())
    bad = times - (TRAIN_TIMES | TEST_TIMES)
    if bad:
        print(f"Warning: unexpected time points {bad}; dropping those cells")
        meta_df = meta_df[meta_df["treatment_time"].isin(TRAIN_TIMES | TEST_TIMES)]
        exp_df = exp_df.loc[meta_df.index]
    train_mask = meta_df["treatment_time"].isin(TRAIN_TIMES)
    test_mask = meta_df["treatment_time"].isin(TEST_TIMES)

    # 5) perturbation_status (pipeline compatibility)
    def _pert_status(row):
        t = row["treatment_time"]
        if t in ("0h", "2h"):
            return "Control"
        if t in ("8h", "10h"):
            return "IFN"
        # 4h, 6h (test): placeholder
        return "IFN"

    meta_df["perturbation_status"] = meta_df.apply(_pert_status, axis=1)
    meta_df["split"] = "train"
    meta_df.loc[test_mask, "split"] = "test"

    # 6) AnnData and save
    adata_full = ad.AnnData(exp_df, obs=meta_df.copy())
    adata_full.obs_names_make_unique()
    compress = None if args.no_gzip else "gzip"

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "fig4_train.h5ad")
    test_path = os.path.join(args.out_dir, "fig4_test.h5ad")

    adata_train = adata_full[train_mask].copy()
    adata_test = adata_full[test_mask].copy()

    adata_train.write_h5ad(train_path, compression=compress)
    adata_test.write_h5ad(test_path, compression=compress)

    print("\nWritten:")
    print(f"  train: {train_path}  -- n_obs={adata_train.n_obs} (0h, 2h, 8h, 10h)")
    print(f"  test:  {test_path}  -- n_obs={adata_test.n_obs} (4h, 6h)")
    print("\nobs columns include: treatment_time, perturbation_status, split, ...")
    print(
        "Setup 1: train with treatment_time as condition; at test, generate 4h/6h and compare to real 4h/6h."
    )


if __name__ == "__main__":
    main()
