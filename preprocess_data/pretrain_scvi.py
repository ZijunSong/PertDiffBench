# scripts/pretrain_scvi.py

import os
import scanpy as sc
import numpy as np
import torch
from scipy import sparse
import scvi

def main(exp_h5ad, save_dir, max_epochs=100):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    print(f"Reading AnnData from {exp_h5ad}...")
    adata_orig = sc.read_h5ad(exp_h5ad)
    print(f"Original AnnData: {adata_orig.n_obs} cells x {adata_orig.n_vars} genes")

    adata = adata_orig.copy()

    if adata.raw is not None:
        print("Found adata.raw; checking whether adata.X differs from adata.raw.X.")
        print("Using adata.raw.X counts for scVI.")
        adata.X = adata.raw.X.copy()
    else:
        print(
            "No adata.raw; assuming adata.X holds raw counts. "
            "Ensure this is correct (non-negative integer counts), or scVI may produce NaNs."
        )

    print(f"Before checks, adata.X type: {type(adata.X)}")
    if sparse.issparse(adata.X):
        print(f"Before checks, adata.X.data type: {type(adata.X.data)}")

    current_X_accessor = adata.X.data if sparse.issparse(adata.X) else adata.X

    try:
        data_for_check = np.asarray(current_X_accessor)
    except Exception as e:
        print(f"Error: cannot convert adata.X to NumPy array for checks: {e}")
        return

    if np.any(data_for_check < 0):
        print("Error: negative values in adata.X; scVI requires non-negative counts.")
        return

    is_already_whole_numbers = np.array_equal(data_for_check, np.round(data_for_check))

    if not is_already_whole_numbers:
        print(
            "Warning: adata.X contains non-integer floats; rounding to int32. "
            "Verify the source is count data."
        )
        if sparse.issparse(adata.X):
            adata.X.data = np.round(data_for_check).astype(np.int32)
        else:
            adata.X = np.round(data_for_check).astype(np.int32)
    elif not np.issubdtype(data_for_check.dtype, np.integer) or data_for_check.dtype != np.int32:
        print("Info: integer values but not int32; converting.")
        if sparse.issparse(adata.X):
            adata.X.data = data_for_check.astype(np.int32)
        else:
            adata.X = data_for_check.astype(np.int32)
    else:
        print("Info: adata.X is non-negative int32 as expected.")

    if sparse.issparse(adata.X):
        print(f"After checks, adata.X.data dtype: {adata.X.data.dtype}")
    else:
        print(f"After checks, adata.X dtype: {adata.X.dtype}")

    print(f"Cells before empty-cell filter: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_genes=1)
    print(f"Cells after empty-cell filter: {adata.n_obs}")

    if adata.n_obs == 0:
        print(
            "Error: all cells removed by min_genes=1. "
            "Dataset may be empty or very sparse."
        )
        return

    print(f"Genes before unexpressed filter: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=1)
    print(f"Genes after unexpressed filter: {adata.n_vars}")

    if adata.n_vars == 0:
        print(
            "Error: all genes removed by min_cells=1. "
            "No genes expressed in any cell."
        )
        return

    print(f"Genes before HVG selection: {adata.n_vars}")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, layer="counts", flavor="seurat_v3")
    print(f"Genes after HVG selection: {adata.n_vars}")

    print("Setting up AnnData for scvi-tools...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Condition"])

    print("Initializing scVI model...")
    vae = scvi.model.SCVI(adata, n_latent=128)

    print(f"Starting training for {max_epochs} epochs...")

    try:
        vae.train(max_epochs=max_epochs, check_val_every_n_epoch=10)
    except ValueError as e:
        print(f"Training ValueError: {e}")
        print("Saving debug AnnData to debug_adata_before_scvi_train.h5ad for inspection.")
        adata.write_h5ad("debug_adata_before_scvi_train.h5ad")
        return
    except Exception as e:
        print(f"Unexpected training error: {e}")
        print("Saving debug AnnData to debug_adata_before_scvi_train.h5ad for inspection.")
        adata.write_h5ad("debug_adata_before_scvi_train.h5ad")
        return

    print(f"Training done. Saving model to {save_dir}...")
    adata.write_h5ad(os.path.join(save_dir, "adata_scvi_for_inference.h5ad"))
    vae.save(save_dir, overwrite=True)

    print(f"scVI model saved to {save_dir}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pretrain scVI model.")
    p.add_argument("--h5ad", default="dataset/scrna_data/scrna_positive.h5ad",
                   help="Input AnnData path (should contain raw counts).")
    p.add_argument("--out", default="checkpoints/scvi_model",
                   help="Directory to save trained scVI model.")
    p.add_argument("--epochs", type=int, default=500,
                   help="Number of training epochs.")
    args = p.parse_args()

    main(args.h5ad, args.out, args.epochs)
