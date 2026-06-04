import anndata
import os
import pandas as pd
import numpy as np

def surgical_inspect_h5ad(file_path: str):
    """
    Read and summarize an h5ad file.

    Fixes inconsistent validation by editing internal _obsp storage directly.

    Args:
        file_path (str): Path to the h5ad file.
    """
    if not os.path.exists(file_path):
        print(f"Error: file not found at '{file_path}'")
        return

    try:
        # Load h5ad into AnnData (validation not run yet).
        adata = anndata.read_h5ad(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    # --- Final fix: operate on internal storage ---
    # Access underlying _obsp dict to bypass public .obsp accessors that trigger errors.
    if hasattr(adata, '_obsp') and adata._obsp is not None:
        n_obs = adata.n_obs
        # Iterate over a copy; do not modify dict while iterating.
        obsp_keys = list(adata._obsp.keys())

        for key in obsp_keys:
            value = adata._obsp[key]
            # Check shape matches (n_obs, n_obs).
            if value.shape[0] != n_obs or value.shape[1] != n_obs:
                print(
                    f"[internal cleanup] Removing ._obsp['{key}'] with shape {value.shape} "
                    f"(expected ({n_obs}, {n_obs}))."
                )
                del adata._obsp[key]

    print("-" * 50)
    print(f"File path: {file_path}")
    print("-" * 50)

    # --- Overview ---
    print("\n[ Overview ]")
    print("AnnData summary:")
    print(adata)

    # --- Data matrix (X) ---
    print("\n" + "=" * 50)
    print("[ Data matrix (X) ]")
    print(f"  Shape: {adata.X.shape}")
    print(f"  Dtype: {adata.X.dtype}")
    try:
        x_preview = adata.X[:5, :5]
        if hasattr(x_preview, "toarray"):
            x_preview = x_preview.toarray()
        print("  First 5x5 preview:")
        with pd.option_context('display.max_rows', 5, 'display.max_columns', 5, 'display.width', 100):
            print(pd.DataFrame(x_preview))
    except Exception as e:
        print(f"  Cannot preview X: {e}")

    # --- Observation metadata (obs) ---
    print("\n" + "=" * 50)
    print("[ obs - cell/sample metadata ]")
    print(f"  Shape: {adata.obs.shape}")
    print(f"  Columns: {list(adata.obs.columns)}")
    print("  First 5 rows:")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 1000):
        print(adata.obs.head())

    # --- Variable metadata (var) ---
    print("\n" + "=" * 50)
    print("[ var - gene/feature metadata ]")
    print(f"  Shape: {adata.var.shape}")
    print(f"  Columns: {list(adata.var.columns)}")
    print("  First 5 rows:")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 1000):
        print(adata.var.head())

    # --- Multidimensional obs annotations (obsm) ---
    print("\n" + "=" * 50)
    print("[ obsm - embeddings, etc. ]")
    if adata.obsm:
        for key, value in adata.obsm.items():
            try:
                print(f"  - '{key}': shape {value.shape}, type {type(value)}")
            except AttributeError:
                print(f"  - '{key}': type {type(value)}")
    else:
        print("  (empty)")

    # --- Unstructured annotations (uns) ---
    print("\n" + "=" * 50)
    print("[ uns - other metadata ]")
    if adata.uns:
        for key, value in adata.uns.items():
            value_repr = repr(value)
            if len(value_repr) > 70:
                value_repr = value_repr[:70] + "..."
            print(f"  - '{key}': type {type(value)}, preview: {value_repr}")
    else:
        print("  (empty)")

    # --- Layers ---
    print("\n" + "=" * 50)
    print("[ layers - alternate matrices ]")
    if adata.layers:
        for key, value in adata.layers.items():
            print(f"  - '{key}': shape {value.shape}, type {type(value)}")
    else:
        print("  (empty)")

    print("\n" + "-" * 50)
    print("Done.")
    print("-" * 50)


if __name__ == "__main__":
    h5ad_file_path = "/share/PertBench/data/add_gaussian_noise_output/task1_train_CD4T_exp_noise_std_0.5.h5ad"
    surgical_inspect_h5ad(h5ad_file_path)
