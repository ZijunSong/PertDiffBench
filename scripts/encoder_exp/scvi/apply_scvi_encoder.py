#!/usr/bin/env python3
"""
Apply a trained scVI encoder to a new AnnData file and write X_scvi.

Example:
    python scripts/encoder_exp/scvi/apply_scvi_encoder.py \
        --data-path data/fig1/raw_task1/task1_valid_CD4T_exp.h5ad \
        --out-h5ad data/fig1/raw_task1/task1_valid_CD4T_exp_with_scvi_latent.h5ad \
        --model-dir checkpoints/scvi_ddpm/scvi_encoder \
        --gpu
"""

import os
import argparse

import scanpy as sc
import scvi
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained scVI model to new AnnData and export X_scvi."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad to be encoded.",
    )
    parser.add_argument(
        "--out-h5ad",
        type=str,
        required=True,
        help="Output .h5ad with adata.obsm['X_scvi'] added.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory of trained scVI model (as saved by model.save).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    args = parser.parse_args()

    print(f"[apply_scvi] Loading AnnData from: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    use_gpu = bool(args.gpu and torch.cuda.is_available())
    device = "cuda" if use_gpu else "cpu"
    print(f"[apply_scvi] Using device: {device}")

    # 这里用 SCVI.load 会自动从保存的模型中恢复 anndata setup 信息
    print(f"[apply_scvi] Loading scVI model from: {os.path.abspath(args.model_dir)}")
    model = scvi.model.SCVI.load(args.model_dir, adata=adata)

    print("[apply_scvi] Computing latent representation...")
    latent = model.get_latent_representation()
    print(f"[apply_scvi] Latent shape: {latent.shape}")

    adata.obsm["X_scvi"] = latent

    out_dir = os.path.dirname(args.out_h5ad)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    adata.write_h5ad(args.out_h5ad)
    print(f"[apply_scvi] ✔ Saved AnnData with X_scvi to: {args.out_h5ad}")
    print("[apply_scvi] Done.")


if __name__ == "__main__":
    main()
