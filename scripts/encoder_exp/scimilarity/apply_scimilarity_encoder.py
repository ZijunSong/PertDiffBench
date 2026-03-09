#!/usr/bin/env python3
"""
Apply pretrained SCimilarity encoder to an AnnData file and write X_scim.

Example:
    python scripts/encoder_exp/scimilarity/apply_scimilarity_encoder.py \
        --data-path data/fig1/raw_task1/task1_train_CD4T_exp.h5ad \
        --out-h5ad samples/encoder_exp/scimilarity_ddpm/task1_train_CD4T_with_scim_latent.h5ad \
        --model-dir /share/PertBench/checkpoints/annotation_model_v1
"""

import os
import argparse

import numpy as np
import scanpy as sc
import torch

from scimilarity import CellAnnotation


def main():
    parser = argparse.ArgumentParser(
        description="Apply pretrained SCimilarity encoder to AnnData and export X_scim."
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
        help="Output .h5ad with adata.obsm['X_scim'] added.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory of pretrained SCimilarity model (has gene_order.tsv, etc.).",
    )
    args = parser.parse_args()

    print(f"[apply_scim] Loading AnnData from: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    print(f"[apply_scim] Loading SCimilarity model from: {os.path.abspath(args.model_dir)}")
    ca = CellAnnotation(model_path=args.model_dir)

    model_genes = list(ca.gene_order)
    model_gene_to_idx = {g: i for i, g in enumerate(model_genes)}
    G_model = len(model_genes)

    data_genes = list(adata.var_names)
    G_data = len(data_genes)

    # build alignment index
    idx_data = []
    idx_model = []
    for j, g in enumerate(data_genes):
        if g in model_gene_to_idx:
            idx_data.append(j)
            idx_model.append(model_gene_to_idx[g])

    if len(idx_data) == 0:
        raise RuntimeError(
            "No overlapping genes between AnnData.var_names and SCimilarity gene_order.\n"
            "Check that gene symbols are compatible."
        )

    print(
        f"[apply_scim] Aligned {len(idx_data)} / {G_data} dataset genes to "
        f"SCimilarity gene space of {G_model} genes."
    )

    # build full matrix X_full: [n_cells, G_model]
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_cells = X.shape[0]
    X_full = np.zeros((n_cells, G_model), dtype=np.float32)
    idx_data = np.array(idx_data, dtype=np.int64)
    idx_model = np.array(idx_model, dtype=np.int64)

    X_full[:, idx_model] = X[:, idx_data]
    print(f"[apply_scim] Full matrix for SCimilarity: shape={X_full.shape}")

    # get embeddings
    print("[apply_scim] Computing SCimilarity embeddings...")
    latent = ca.get_embeddings(X_full)
    print(f"[apply_scim] Latent shape: {latent.shape}")

    adata.obsm["X_scim"] = latent

    out_dir = os.path.dirname(args.out_h5ad)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    adata.write_h5ad(args.out_h5ad)
    print(f"[apply_scim] ✔ Saved AnnData with X_scim to: {args.out_h5ad}")
    print("[apply_scim] Done.")


if __name__ == "__main__":
    main()
