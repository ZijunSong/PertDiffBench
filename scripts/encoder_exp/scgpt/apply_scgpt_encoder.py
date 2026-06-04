#!/usr/bin/env python3
"""
Apply pretrained scGPT encoder to an AnnData file and store cell embeddings
in adata.obsm['X_scgpt'] with simple resume-by-file logic.

This script tries to be robust to different scGPT versions:
- embed_data may return:
    * an AnnData with embeddings in .obsm
    * a tuple (AnnData, embeddings)
    * embeddings as a numpy/torch array
"""

import os
import argparse
import numpy as np
import inspect
import scanpy as sc


def detect_gene_col(adata):
    """
    Try to find a reasonable gene name column in adata.var.

    If none of the common names exist, we create a new column 'feature_name'
    from adata.var_names, which is compatible with many scGPT examples.
    """
    candidates = ["feature_name", "gene_symbol", "gene_name", "Gene", "genes"]
    for col in candidates:
        if col in adata.var.columns:
            print(f"[scGPT] Using adata.var['{col}'] as gene_col.")
            return col

    adata.var["feature_name"] = adata.var_names
    print(
        "[scGPT] No common gene_col found in adata.var. "
        "Created adata.var['feature_name'] from adata.var_names."
    )
    return "feature_name"


def _is_anndata(obj):
    """Lightweight check if obj looks like an AnnData."""
    return hasattr(obj, "obs") and hasattr(obj, "var") and hasattr(obj, "obsm")


def run_scgpt_encoder_on_adata(adata, ckpt_dir: str, device: str = "cuda"):
    """
    Run scGPT's embed_data() on an AnnData and return (adata_with_latent, latent_array).

    This version is tailored to the embed_data implementation you showed:

        def embed_data(
            adata_or_file,
            model_dir,
            gene_col="feature_name",
            max_length=1200,
            batch_size=64,
            obs_to_save=None,
            device="cuda",
            use_fast_transformer=True,
            return_new_adata=False,
        ) -> AnnData:

    - If return_new_adata=False (default), it:
        * writes embeddings to adata.obsm["X_scGPT"]
        * returns the same AnnData
    - If return_new_adata=True, it:
        * returns a NEW AnnData with X = cell_embeddings (no obsm["X_scGPT"])
    """
    try:
        from scgpt.tasks.cell_emb import embed_data
    except ImportError as e:
        raise ImportError(
            "scgpt is not installed or not found in your environment. "
            "Please install it via `pip install scgpt` (or conda/mamba)."
        ) from e

    # 1) gene_col must exist, else embed_data will assert
    gene_col = detect_gene_col(adata)

    # here embed_data and must cell_type, to , 
    if "cell_type" not in adata.obs.columns:
        print("[scGPT] adata.obs has no 'cell_type' column. Creating a dummy one.")
        adata.obs["cell_type"] = "unknown"

    print(f"[scGPT] Running embed_data() with model_dir={ckpt_dir}, device={device}")
    # 2) key: hereweexplicit using return_new_adata=False
    new_adata = embed_data(
        adata_or_file=adata,
        model_dir=ckpt_dir,
        gene_col=gene_col,
        max_length=1200,
        batch_size=64,
        obs_to_save=None,
        device=device,
        use_fast_transformer=True,
        return_new_adata=False, # <---- key
    )

    # , when new_adata adata 
    if "X_scGPT" not in new_adata.obsm:
        # hereno, ckpt_dir or scGPT 
        raise RuntimeError(
            "scGPT embed_data() finished but new_adata.obsm['X_scGPT'] not found. "
            "Please check your checkpoint directory and scGPT version."
        )

    latent = np.asarray(new_adata.obsm["X_scGPT"], dtype=np.float32)
    print(f"[scGPT] Got latent from new_adata.obsm['X_scGPT'], shape={latent.shape}")

    if latent.shape[0] != new_adata.n_obs:
        raise ValueError(
            f"Latent shape {latent.shape} does not match n_obs={new_adata.n_obs}."
        )

    # 3) as DDPM pipeline , outside key
    new_adata.obsm["X_scgpt"] = latent
    print(
        "[scGPT] Stored cell embeddings in new_adata.obsm['X_scGPT'] "
        "and new_adata.obsm['X_scgpt']."
    )

    return new_adata, latent

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply scGPT encoder to AnnData and store embeddings in "
            "obsm['X_scgpt'] with resume-by-file logic."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad.",
    )
    parser.add_argument(
        "--out-h5ad",
        type=str,
        required=True,
        help="Output .h5ad with adata.obsm['X_scgpt'] filled.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Directory containing pretrained scGPT checkpoint (and vocab, config, etc).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for scGPT, e.g. 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out_h5ad)

    # resume-by-file: if output exists and has X_scgpt, skip
    if os.path.exists(out_path):
        adata_existing = sc.read_h5ad(out_path)
        if "X_scgpt" in adata_existing.obsm:
            print(
                f"[scGPT] Found existing output with 'X_scgpt' at {out_path}. "
                f"Skipping encoding."
            )
            return
        else:
            print(
                f"[scGPT] Output file {out_path} exists but has no 'X_scgpt'. "
                f"Will recompute embeddings."
            )

    print(f"[scGPT] Loading input AnnData from: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    adata_with_latent, latent = run_scgpt_encoder_on_adata(
        adata, ckpt_dir=args.ckpt_dir, device=args.device
    )
    print(f"[scGPT] Latent shape: {latent.shape}")

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    adata_with_latent.write_h5ad(out_path)
    print(f"[scGPT] ✔ Saved AnnData with 'X_scgpt' to: {out_path}")


if __name__ == "__main__":
    main()
