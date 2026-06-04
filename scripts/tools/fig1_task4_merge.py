# merge_to_h5ad.py

import sys
import pandas as pd
import anndata as ad

def merge_csv_to_h5ad(meta_file, exp_file, output_file):
    """
    cell data (meta) geneexpressedmatrix (exp) CSV file and H5AD file.

    args:
    meta_file (str): cell data CSV filepath ( : cell, cols: data ).
    exp_file (str): geneexpressedmatrix CSV filepath ( : gene, cols: cell).
    output_file (str): output H5AD filepath.
    """
    # --- 1. Load data ---
    print(f"--- Readingfile ---")
    try:
        # Assume filefirstcols indexcols
        print(f" datafile: {meta_file}")
        meta_df = pd.read_csv(meta_file, index_col=0)
        
        print(f" expressedmatrixfile: {exp_file}")
        exp_df = pd.read_csv(exp_file, index_col=0)
    except FileNotFoundError as e:
        print(f" : file found - {e}")
        sys.exit(1) # andreturn 
    except Exception as e:
        print(f" filewhenoccurred : {e}")
        sys.exit(1)

    print("\n--- data ---")
    print(f" data (meta) shape: {meta_df.shape}")
    print(f"expressedmatrix (exp) shape: {exp_df.shape}")

    # --- 2. Align data (important) ---
    print("\n--- inaligncellID ---")
    # found data expressedmatrix cellID
    common_cells = meta_df.index.intersection(exp_df.columns)

    if len(common_cells) == 0:
        print(" : expressedmatrixcellID (colsname) datacellID (index) no .")
        print(" check fileinside whether .")
        sys.exit(1)
    
    if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.columns):
        print(" : and allcell in file on.only using cell.")
        print(f" {len(common_cells)} cellcontainin file .")

    # based on cellIDfilterand , Ensure 
    meta_df_aligned = meta_df.loc[common_cells]
    exp_df_aligned = exp_df[common_cells]

    # --- 3. Create AnnData ---
    # AnnData must cell x gene (obs x var) expressedmatrix.
    # we exp_df_aligned gene x cell, tomusttranspose (.T).
    print("\n--- Creating AnnData object ---")
    adata = ad.AnnData(X=exp_df_aligned.T,  # transposeexpressedmatrix
                       obs=meta_df_aligned) # obs (observations) cell data
    
    # anndata will fromtransposeafterexpressedmatrixcols var (variables, gene data).

    # --- 4. Validate object ---
    print("\n--- AnnData objectbatch info ---")
    print(adata)
    print(f" value (cell) data (obs) cols: {list(adata.obs.columns)}")
    print(f" amount (gene) index (var) name : {list(adata.var.index[:5])}")

    # --- 5. Save H5AD ---
    print(f"\n--- Saving to H5AD file ---")
    try:
        adata.write_h5ad(output_file)
        print(f"\n✅ !data andand to '{output_file}' file .")
    except Exception as e:
        print(f" H5AD filewhenoccurred : {e}")
        sys.exit(1)

if __name__ == '__main__':
    # checkCLIargscountamountwhethercorrect
    if len(sys.argv) != 4:
        print("using : python merge_to_h5ad.py <meta_csv_path> <exp_csv_path> <output_h5ad_path>")
        print(" : python merge_to_h5ad.py meta.csv exp.csv merged_data.h5ad")
        sys.exit(1)

    # fromCLIgetfilename
    meta_csv_path = sys.argv[1]
    exp_csv_path = sys.argv[2]
    output_h5ad_path = sys.argv[3]

    # call count and 
    merge_csv_to_h5ad(meta_csv_path, exp_csv_path, output_h5ad_path)