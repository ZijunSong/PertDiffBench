# merge_to_h5ad.py

import sys
import pandas as pd
import anndata as ad

def merge_csv_to_h5ad(meta_file, exp_file, output_file):
    """
    Merge cell metadata (meta) and gene expression matrix (exp) CSVs into one H5AD file.

    Args:
        meta_file (str): Cell metadata CSV (rows: cells, columns: metadata features).
        exp_file (str): Expression matrix CSV (rows: genes, columns: cells).
        output_file (str): Output H5AD path.
    """
    print("--- Reading files ---")
    try:
        print(f"Reading metadata: {meta_file}")
        meta_df = pd.read_csv(meta_file, index_col=0)

        print(f"Reading expression matrix: {exp_file}")
        exp_df = pd.read_csv(exp_file, index_col=0)
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    print("\n--- Data preview ---")
    print(f"metadata (meta) shape: {meta_df.shape}")
    print(f"expression matrix (exp) shape: {exp_df.shape}")

    print("\n--- Aligning cell IDs ---")
    common_cells = meta_df.index.intersection(exp_df.columns)

    if len(common_cells) == 0:
        print("Error: no overlap between expression column names and metadata index.")
        print("Check that file contents match.")
        sys.exit(1)

    if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.columns):
        print("Warning: not all cells match in both files; using intersection only.")
        print(f"Using {len(common_cells)} common cells in final file.")

    meta_df_aligned = meta_df.loc[common_cells]
    exp_df_aligned = exp_df[common_cells]

    # exp_df_aligned is gene x cell; AnnData expects cell x gene
    print("\n--- Creating AnnData object ---")
    adata = ad.AnnData(
        X=exp_df_aligned.T,
        obs=meta_df_aligned
    )

    print("\n--- AnnData summary ---")
    print(adata)
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var index example: {list(adata.var.index[:5])}")

    print("\n--- Saving H5AD ---")
    try:
        adata.write_h5ad(output_file)
        print(f"\nDone: merged data saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving H5AD: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python merge_to_h5ad.py <meta_csv_path> <exp_csv_path> <output_h5ad_path>")
        print("Example: python merge_to_h5ad.py meta.csv exp.csv merged_data.h5ad")
        sys.exit(1)

    meta_csv_path = sys.argv[1]
    exp_csv_path = sys.argv[2]
    output_h5ad_path = sys.argv[3]

    merge_csv_to_h5ad(meta_csv_path, exp_csv_path, output_h5ad_path)
