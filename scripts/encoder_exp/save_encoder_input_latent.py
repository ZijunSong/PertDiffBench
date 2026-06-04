#!/usr/bin/env python3
"""Save encoder input latent matrix (adata.obsm) for train/test analysis."""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export DDPM encoder-input latent matrix from AnnData.obsm to .npy (+ optional obs names)."
    )
    p.add_argument("--h5ad", required=True, help="AnnData .h5ad that already contains obsm[latent-key].")
    p.add_argument("--latent-key", required=True, help="obsm key, e.g. X_scvi, X_scgpt.")
    p.add_argument("--out-npy", required=True, help="Output path (use .npy suffix).")
    p.add_argument(
        "--out-obs-names",
        default=None,
        help="Optional text file: one cell barcode / obs name per line, aligned with matrix rows.",
    )
    args = p.parse_args()

    adata = ad.read_h5ad(args.h5ad)
    if args.latent_key not in adata.obsm:
        raise SystemExit(
            f"Missing obsm['{args.latent_key}'] in {args.h5ad}. Keys: {list(adata.obsm.keys())}"
        )
    x = np.asarray(adata.obsm[args.latent_key], dtype=np.float32)
    np.save(args.out_npy, x)
    print(f"[save_encoder_input_latent] {args.h5ad} obsm['{args.latent_key}'] shape={x.shape} -> {args.out_npy}")
    if args.out_obs_names:
        with open(args.out_obs_names, "w", encoding="utf-8") as f:
            for name in adata.obs_names.astype(str):
                f.write(name + "\n")
        print(f"[save_encoder_input_latent] obs names ({len(adata)}) -> {args.out_obs_names}")


if __name__ == "__main__":
    main()
