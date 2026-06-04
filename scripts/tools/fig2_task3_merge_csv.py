# create_h5ad.py

import pandas as pd
import anndata as ad
import argparse
import sys
import os

def verify_h5ad(file_path):
    """
     .h5ad fileand keybatch infoto validate.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: validate , file '{file_path}' exist.")
        return
        
    print("\n--- [validation step] ---")
    print(f"INFO: in file '{file_path}' checks...")
    
    try:
        adata_check = ad.read_h5ad(file_path)
        
        print("\n1. AnnData object must:")
        print(adata_check)
        
        print("\n2. cell data (obs) before5 :")
        print(adata_check.obs.head())
        
        print("\n3. gene data (var) before5 :")
        print(adata_check.var.head())

        print(f"\nINFO: validate .data contain {adata_check.n_obs} cell {adata_check.n_vars} gene.")
        print("--- [validate ] ---\n")

    except Exception as e:
        print(f"ERROR: validateprocess orcheckfile '{file_path}' whenoccurred : {e}")


def create_h5ad(meta_path, exp_path, output_path):
    """
     meta.csv exp.csv file and h5ad file.

    Args:
        meta_path (str): meta.csv filepath (cell data).
        exp_path (str): exp.csv filepath (expressedmatrix, as cell x gene).
        output_path (str): output .h5ad file path.
    """
    try:
        # --- step 1: cell data (obs) ---
        print(f"INFO: Loading from '{meta_path}' data...")
        # CSVfirstcols cellID, we asindexcols
        meta_df = pd.read_csv(meta_path, index_col=0)

        # --- step 2: based on requiresadd 'Cell.Type' cols ---
        # asallcell 'Cell.Type' cols valueas 'species'
        meta_df['Cell.Type'] = 'species'
        print("INFO: 'Cell.Type' colsand valueas 'species'.")

        # --- step 3: geneexpressedmatrix (X) ---
        print(f"INFO: Loading from '{exp_path}' expressedmatrix...")
        # CSVfirstcols cellID, asindex; cols genename
        exp_df = pd.read_csv(exp_path, index_col=0)

        # --- step 4: EnsurecellID (index)align ---
        # AnnData requires obs X index .
        # we using fileindex Ensuredataalign.
        common_cells = meta_df.index.intersection(exp_df.index)

        if len(common_cells) == 0:
            print("ERROR: datafile expressedmatrixfile no cellID, cannot .")
            sys.exit(1)
        
        # filecell , 
        if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.index):
            print(f"WARNING: and allcell whenexist file .will use {len(common_cells)} cellrun and.")
        
        # based on cellID align DataFrame
        meta_df_aligned = meta_df.loc[common_cells]
        exp_df_aligned = exp_df.loc[common_cells]

        # --- step 5: AnnData object ---
        print("INFO: Creating AnnData object...")
        # X: expressedmatrix (cell x gene)
        # obs: cell data
        # var: gene data (will fromexpressedmatrixcolsname )
        adata = ad.AnnData(
            X=exp_df_aligned,
            obs=meta_df_aligned
        )

        # --- step 6: as h5ad file ---
        print(f"INFO: Writing AnnData object to '{output_path}'...")
        # using gzip canto file 
        adata.write_h5ad(output_path, compression="gzip")

        print("\n🎉 handledone!")
        
        # --- step 7: validate file ---
        verify_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: file found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: handleprocess occurred - {e}")
        sys.exit(1)


if __name__ == '__main__':
    # --- argparse setup ---
    parser = argparse.ArgumentParser(
        description="cell data (meta.csv) geneexpressedmatrix (exp.csv) andas h5ad file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'meta_file', 
        help='input dataCSVfilepath (e.g.: meta.csv).'
    )
    parser.add_argument(
        'exp_file', 
        help='inputexpressedmatrixCSVfilepath (e.g.: exp.csv).'
    )
    parser.add_argument(
        'output_file', 
        help='output .h5ad filepath (e.g.: output.h5ad).'
    )

    # parseCLIargs
    args = parser.parse_args()

    # call count convert
    create_h5ad(args.meta_file, args.exp_file, args.output_file)
