#!/usr/bin/env python3
"""
Attach scFoundation cell embeddings back to a PertBench AnnData file.

Usage example:

    python scripts/encoder_exp/scfoundation/attach_scfoundation_embedding.py \
        --orig-h5ad data/fig1/raw_task1/task1_train_CD4T_exp.h5ad \
        --pre-h5ad /path/to/scFoundation-main/preprocessing/output/preprocessed_task1_train_CD4T_exp.h5ad \
        --embedding-npy /path/to/scFoundation-main/model/output/single_cell_data/task1_train_CD4T_exp_cell_embedding.npy \
        --out-h5ad samples/encoder_exp/scfoundation_ddpm/task1_train_CD4T_with_scf_latent.h5ad \
        --obsm-key X_scfoundation
"""

import os
import argparse

import numpy as np
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(
        description="Attach scFoundation embeddings to a PertBench AnnData file."
    )
    parser.add_argument("--orig-h5ad", required=True, help="Original PertBench AnnData (gene space & obs you want to keep).")
    parser.add_argument("--pre-h5ad", required=True, help="Preprocessed AnnData used for scFoundation get_embedding.py.")
    parser.add_argument("--embedding-npy", required=True, help="Path to scFoundation cell embedding .npy file.")
    parser.add_argument("--out-h5ad", required=True, help="Output AnnData with obsm[obsm_key].")
    parser.add_argument(
        "--obsm-key",
        default="X_scfoundation",
        help="Key name to store embeddings in adata.obsm (default: X_scfoundation).",
    )
    args = parser.parse_args()

    print(f"[attach_scf] Loading original AnnData from: {os.path.abspath(args.orig_h5ad)}")
    adata_orig = sc.read_h5ad(args.orig_h5ad)

    print(f"[attach_scf] Loading preprocessed AnnData from: {os.path.abspath(args.pre_h5ad)}")
    adata_pre = sc.read_h5ad(args.pre_h5ad)

    print(f"[attach_scf] Loading embeddings from: {os.path.abspath(args.embedding_npy)}")
    emb = np.load(args.embedding_npy)
    print(f"[attach_scf] Embedding shape: {emb.shape}")

    # 基本 sanity check：cell 数量要一致
    if adata_pre.n_obs != emb.shape[0]:
        raise ValueError(
            f"Mismatch between preprocessed AnnData cells ({adata_pre.n_obs}) "
            f"and embedding rows ({emb.shape[0]})."
        )

    # 再确保 cell 名称是一一对应的，如果不对应就按名称对齐
    if np.array_equal(adata_orig.obs_names.values, adata_pre.obs_names.values):
        print("[attach_scf] obs_names are aligned between orig and preprocessed AnnData.")
        emb_aligned = emb
    else:
        print("[attach_scf] obs_names are NOT aligned, aligning by index name intersection...")
        # cells present in both
        common_cells = adata_orig.obs_names.intersection(adata_pre.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping cells between orig and preprocessed AnnData.")

        # build name -> row index mapping in preprocessed AnnData
        # BEFORE any subsetting，embedding 的行顺序与 adata_pre.obs_names 一致
        cell_to_row = {cell: i for i, cell in enumerate(adata_pre.obs_names)}

        # determine embedding row indices in the order of `common_cells`
        idx = [cell_to_row[c] for c in common_cells]

        emb_aligned = emb[idx, :]

        # subset original AnnData to these cells，顺序与 common_cells 一致
        adata_orig = adata_orig[common_cells].copy()

    # 把 embedding 塞到 obsm 里
    adata_orig.obsm[args.obsm_key] = emb_aligned

    out_dir = os.path.dirname(args.out_h5ad)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    adata_orig.write_h5ad(args.out_h5ad)
    print(f"[attach_scf] ✔ Saved AnnData with {args.obsm_key} to: {args.out_h5ad}")


if __name__ == "__main__":
    main()
