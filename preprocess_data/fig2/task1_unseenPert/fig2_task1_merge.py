import pandas as pd
import anndata as ad
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Merge expression (exp) and metadata (meta) CSV files into one H5AD file.'
    )

    parser.add_argument('--exp', type=str, required=True, help='Expression matrix CSV (cells x genes).')
    parser.add_argument('--meta', type=str, required=True, help='Metadata CSV path.')
    parser.add_argument('--output', type=str, required=True, help='Output H5AD path.')

    args = parser.parse_args()

    print(f"Loading expression matrix from {args.exp}...")
    exp_df = pd.read_csv(args.exp, index_col=0)

    print(f"Loading metadata from {args.meta}...")
    meta_df = pd.read_csv(args.meta, index_col=0)

    if 'celltype' in meta_df.columns:
        meta_df.rename(columns={'celltype': 'Cell.Type'}, inplace=True)

    adata = ad.AnnData(exp_df)

    print("Matching cell IDs and attaching metadata...")
    common_cells = exp_df.index.intersection(meta_df.index)
    adata = adata[common_cells, :].copy()
    adata.obs = meta_df.loc[common_cells]

    print(f"Saving merged data to {args.output}...")
    adata.write_h5ad(args.output, compression='gzip')

    print("\nDone.")
    print("\nMerged AnnData:")
    print(adata)

if __name__ == '__main__':
    main()
