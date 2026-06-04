import pandas as pd
import anndata as ad
import argparse # argparse 

def main():
    # --- 1. argparse setup ---
    parser = argparse.ArgumentParser(description='expressedmatrix (exp) data (meta) CSV file and H5AD file.')
    
    # define needargs
    parser.add_argument('--exp', type=str, required=True, help='inputexpressedmatrix CSV filepath (cells x genes).')
    parser.add_argument('--meta', type=str, required=True, help='input data CSV filepath.')
    parser.add_argument('--output', type=str, required=True, help='output H5AD filepath.')
    
    # parse args
    args = parser.parse_args()

    # --- 2. Load data ---
    print(f"Loading from {args.exp} expressedmatrix...")
    # index_col=0 usingfirstcols as index (cell ID)
    exp_df = pd.read_csv(args.exp, index_col=0)

    print(f"Loading from {args.meta} data...")
    # , usingfirstcols as index
    meta_df = pd.read_csv(args.meta, index_col=0)
    
    if 'celltype' in meta_df.columns:
        meta_df.rename(columns={'celltype': 'Cell.Type'}, inplace=True)

    # --- 3. Create AnnData ---
    # AnnData expectsinput ( value × amount), (cell × gene)
    adata = ad.AnnData(exp_df)

    # --- 4. Match metadata ---
    print(" in cell ID andadd data...")
    # asEnsure , we keep whenexist expressedmatrix data cell
    # andperexpressedmatrix for datarun 
    common_cells = exp_df.index.intersection(meta_df.index)
    adata = adata[common_cells, :].copy()
    adata.obs = meta_df.loc[common_cells]

    # --- 5. Save h5ad ---
    print(f"Writing andafterdata to {args.output}...")
    adata.write_h5ad(args.output, compression='gzip')

    print("\nhandledone! 🎉")
    print("\n AnnData objectbatch info under:")
    print(adata)

if __name__ == '__main__':
    main()