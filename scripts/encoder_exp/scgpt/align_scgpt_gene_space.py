#!/usr/bin/env python3
"""
Align gene space between scGPT-encoded train/valid AnnData.

We:
  1) Read train_with_latent.h5ad and valid_with_latent.h5ad
  2) Compute the intersection of gene names
  3) Subset BOTH AnnData to this common gene set, with the same order as TRAIN
  4) Save aligned versions to new paths

After this, both .h5ad have:
  - the same number of genes (n_vars)
  - the same var_names and order
"""

import os
import argparse
import scanpy as sc
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Align gene space between scGPT train/valid AnnData."
    )
    parser.add_argument(
        "--train-in",
        required=True,
        help="Input train h5ad with scGPT latent (before alignment).",
    )
    parser.add_argument(
        "--valid-in",
        required=True,
        help="Input valid h5ad with scGPT latent (before alignment).",
    )
    parser.add_argument(
        "--train-out",
        required=True,
        help="Output train h5ad with aligned genes.",
    )
    parser.add_argument(
        "--valid-out",
        required=True,
        help="Output valid h5ad with aligned genes.",
    )
    args = parser.parse_args()

    print(f"[ALIGN] Reading train from: {os.path.abspath(args.train_in)}")
    adata_train = sc.read_h5ad(args.train_in)

    print(f"[ALIGN] Reading valid from: {os.path.abspath(args.valid_in)}")
    adata_valid = sc.read_h5ad(args.valid_in)

    genes_train = np.array(adata_train.var_names)
    genes_valid = np.array(adata_valid.var_names)

    print(f"[ALIGN] #genes in TRAIN: {len(genes_train)}")
    print(f"[ALIGN] #genes in VALID: {len(genes_valid)}")

    # intersection, but keep TRAIN order
    genes_train_set = set(genes_train)
    genes_valid_set = set(genes_valid)
    common_genes = [g for g in genes_train if g in genes_valid_set]

    if len(common_genes) == 0:
        raise ValueError(
            "No common genes between train and valid after scGPT filtering. "
            "Please check your datasets."
        )

    print(f"[ALIGN] #common genes: {len(common_genes)}")

    # subset both AnnData to common genes, in the same order
    adata_train_aligned = adata_train[:, common_genes].copy()
    adata_valid_aligned = adata_valid[:, common_genes].copy()

    print(
        f"[ALIGN] Aligned TRAIN: n_obs={adata_train_aligned.n_obs}, "
        f"n_vars={adata_train_aligned.n_vars}"
    )
    print(
        f"[ALIGN] Aligned VALID: n_obs={adata_valid_aligned.n_obs}, "
        f"n_vars={adata_valid_aligned.n_vars}"
    )

    # save
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.valid_out), exist_ok=True)

    adata_train_aligned.write_h5ad(args.train_out)
    adata_valid_aligned.write_h5ad(args.valid_out)

    print(f"[ALIGN] ✔ Saved aligned TRAIN to: {args.train_out}")
    print(f"[ALIGN] ✔ Saved aligned VALID to: {args.valid_out}")


if __name__ == "__main__":
    main()
