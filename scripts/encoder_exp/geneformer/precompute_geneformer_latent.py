#!/usr/bin/env python3
# scripts/encoder_exp/precompute_geneformer_latent.py

import os
import argparse

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from geneformer_encoder import GeneformerEncoder

def prepare_h5ad_for_geneformer(h5ad_path: str):
    """
    Make sure the input h5ad has the minimum fields Geneformer expects:

    1) adata.var["ensembl_id"]
    2) adata.obs["n_counts"]

    只有在发生实际修改时才写回文件，避免 HDF5 文件锁问题。
    """

    print(f"[prepare_h5ad_for_geneformer] Checking {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    changed = False

    # ------------------- var: ensembl_id -------------------
    var = adata.var
    if "ensembl_id" in var.columns:
        print("[prepare_h5ad_for_geneformer] Found var['ensembl_id'], keep as-is.")
    else:
        print("[prepare_h5ad_for_geneformer] var['ensembl_id'] not found, trying to infer...")
        candidate_cols = [
            "gene_id",
            "gene_ids",
            "ensembl",
            "ensembl_ids",
            "gene",
            "gene_name",
            "symbol",
        ]
        used_col = None
        for col in candidate_cols:
            if col in var.columns:
                used_col = col
                break

        if used_col is not None:
            print(f"[prepare_h5ad_for_geneformer] Using var['{used_col}'] to create var['ensembl_id'].")
            var["ensembl_id"] = var[used_col].astype(str)
        else:
            print("[prepare_h5ad_for_geneformer] No obvious ID column found; "
                  "using var.index as ensembl_id (may not be true Ensembl IDs).")
            var["ensembl_id"] = var.index.astype(str)
        changed = True

    # ------------------- obs: n_counts -------------------
    obs = adata.obs
    if "n_counts" in obs.columns:
        print("[prepare_h5ad_for_geneformer] Found obs['n_counts'], keep as-is.")
    else:
        print("[prepare_h5ad_for_geneformer] obs['n_counts'] not found, computing from X...")
        X = adata.X
        if issparse(X):
            counts = np.array(X.sum(1)).ravel()
        else:
            counts = X.sum(1)
        obs["n_counts"] = counts.astype(np.float32)
        changed = True

    adata.var = var
    adata.obs = obs

    if changed:
        adata.write_h5ad(h5ad_path)
        print(f"[prepare_h5ad_for_geneformer] Updated file written back to {h5ad_path}")
    else:
        print("[prepare_h5ad_for_geneformer] No changes needed; skip writing.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geneformer-root", type=str, required=True,
                        help="Path to cloned Geneformer repo (with config.json, model.safetensors).")
    parser.add_argument("--input-h5ad", type=str, required=True,
                        help="Input raw-counts h5ad file.")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory for intermediate .dataset, embeddings, and encoded h5ad.")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Prefix for outputs, e.g., task1_train_CD4T.")
    parser.add_argument("--model-version", type=str, default="V2",
                        help="Geneformer model version: 'V1' or 'V2'. Used only if supported by your installation.")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip steps whose outputs already exist.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 先保证 input h5ad 满足 Geneformer 的基本字段要求
    prepare_h5ad_for_geneformer(args.input_h5ad)

    encoder = GeneformerEncoder(
        geneformer_root=args.geneformer_root,
        model_version=args.model_version,
        nproc=args.nproc,
    )

    # 1) tokenize
    dataset_path = encoder.tokenize_h5ad(
        input_h5ad=args.input_h5ad,
        output_dir=os.path.join(args.out_dir, "datasets"),
        output_prefix=args.prefix,
        custom_attr_name_dict={},  # 若需要保留 obs label，可在这里加
        resume=args.resume,
    )

    # 2) extract embeddings
    emb_csv = encoder.extract_embeddings(
        dataset_path=dataset_path,
        output_dir=os.path.join(args.out_dir, "embeddings"),
        output_prefix=args.prefix,
        model_dir=None,             # None -> use geneformer_root
        emb_mode="cell",
        emb_layer=-1,
        max_ncells=None,
        resume=args.resume,
    )

    # 3) merge back to h5ad
    encoded_h5ad = encoder.write_embeddings_to_h5ad(
        input_h5ad=args.input_h5ad,
        emb_csv=emb_csv,
        output_h5ad=os.path.join(args.out_dir, f"{args.prefix}_geneformer_latent.h5ad"),
        obsm_key="X_geneformer",
        resume=args.resume,
    )

    print(f"[main] Finished. Encoded h5ad at: {encoded_h5ad}")


if __name__ == "__main__":
    main()
