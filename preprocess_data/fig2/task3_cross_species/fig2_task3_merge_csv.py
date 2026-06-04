# create_h5ad.py

import pandas as pd
import anndata as ad
import argparse
import sys
import os

def verify_h5ad(file_path):
    """Read a .h5ad file and print key info for validation."""
    if not os.path.exists(file_path):
        print(f"ERROR: validation failed, file '{file_path}' does not exist.")
        return

    print("\n--- [validation] ---")
    print(f"INFO: re-reading saved file '{file_path}' for checks...")

    try:
        adata_check = ad.read_h5ad(file_path)

        print("\n1. AnnData summary:")
        print(adata_check)

        print("\n2. cell metadata (obs) first 5 rows:")
        print(adata_check.obs.head())

        print("\n3. gene metadata (var) first 5 rows:")
        print(adata_check.var.head())

        print(f"\nINFO: validation OK. Dataset has {adata_check.n_obs} cells and {adata_check.n_vars} genes.")
        print("--- [validation end] ---\n")

    except Exception as e:
        print(f"ERROR: validation failed reading '{file_path}': {e}")


def create_h5ad(meta_path, exp_path, output_path):
    """
    Merge meta.csv and exp.csv into one h5ad file.

    Args:
        meta_path (str): meta.csv path (cell metadata).
        exp_path (str): exp.csv path (expression matrix, cell x gene).
        output_path (str): output .h5ad save path.
    """
    try:
        print(f"INFO: reading metadata from '{meta_path}'...")
        meta_df = pd.read_csv(meta_path, index_col=0)

        meta_df['Cell.Type'] = 'species'
        print("INFO: created 'Cell.Type' column with value 'species'.")

        print(f"INFO: reading expression matrix from '{exp_path}'...")
        exp_df = pd.read_csv(exp_path, index_col=0)

        common_cells = meta_df.index.intersection(exp_df.index)

        if len(common_cells) == 0:
            print("ERROR: no common cell IDs between metadata and expression files.")
            sys.exit(1)

        if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.index):
            print(f"WARNING: not all cells in both files; using {len(common_cells)} common cells.")

        meta_df_aligned = meta_df.loc[common_cells]
        exp_df_aligned = exp_df.loc[common_cells]

        print("INFO: creating AnnData object...")
        adata = ad.AnnData(
            X=exp_df_aligned,
            obs=meta_df_aligned
        )

        print(f"INFO: saving AnnData to '{output_path}'...")
        adata.write_h5ad(output_path, compression="gzip")

        print("\nDone.")

        verify_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: processing failed - {e}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge cell metadata (meta.csv) and expression (exp.csv) into h5ad.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('meta_file', help='Input metadata CSV (e.g. meta.csv).')
    parser.add_argument('exp_file', help='Input expression CSV (e.g. exp.csv).')
    parser.add_argument('output_file', help='Output .h5ad path (e.g. output.h5ad).')

    args = parser.parse_args()
    create_h5ad(args.meta_file, args.exp_file, args.output_file)
