import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

# Input: raw CSV (same style as fig1_task1: data_ori/fig1/raw_task*)
data_dir = '/data/ppnm/data/PertDiffBench/data_ori/fig1/raw_task3/'
# Output: processed H5AD
out_dir = '/data/ppnm/data/PertDiffBench/data/fig1_task3'

print(f"Processing CSV files in: {data_dir}...")
print(f"H5AD output dir: {out_dir}...")

csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"Error: no CSV files found in {data_dir}.")
    exit()

os.makedirs(out_dir, exist_ok=True)

for csv_file in csv_files:
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(out_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- Processing file: {csv_file} ---")

    try:
        # Read CSV: skip first column (row id); use second column 'index' as DataFrame index
        df = pd.read_csv(csv_file, index_col=1)
        print("CSV read OK.")
        print(f"Raw data shape (cell x gene): {df.shape}")
    except FileNotFoundError:
        print(f"Error: file not found {csv_file}. Check path.")
        continue
    except Exception as e:
        print(f"Error reading CSV {csv_file}: {e}")
        continue

    # Data is already cell x gene; no legacy metadata rows to transpose
    df_expression = df
    print("Confirmed cell x gene layout; no transpose needed.")

    adata = ad.AnnData(df_expression)
    print("AnnData object created.")

    print("Adding metadata to adata.obs ...")

    try:
        cell_type = base_name.split('_')[2]
        adata.obs['Cell.Type'] = cell_type
        print(f"Added 'Cell.Type' column: {cell_type}.")
    except IndexError:
        print("Warning: could not parse 'Cell.Type' from filename; using 'unknown'.")
        adata.obs['Cell.Type'] = 'unknown'

    conditions = [
        adata.obs.index.str.endswith('control'),
        adata.obs.index.str.endswith('stimulated')
    ]
    choices = ['Control', 'IFN']

    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')
    print("Added 'perturbation_status' from cell ID suffix.")

    print("Metadata added.")

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(adata.X)

    try:
        adata.write(h5ad_file)
        print(f"Saved (with metadata) to: {h5ad_file}")
    except Exception as e:
        print(f"Error saving H5AD: {e}")

    print("--- Validation ---")
    print(f"Re-reading saved H5AD: {h5ad_file}")
    try:
        adata_loaded = ad.read_h5ad(h5ad_file)
        print("Reload OK.")
        print("Metadata after reload (first 5 rows):")
        print(adata_loaded.obs.head())
        print("Gene info (adata.var, first 5 rows):")
        print(adata_loaded.var.head())
    except Exception as e:
        print(f"Error re-reading H5AD {h5ad_file}: {e}")

print("\n------ All file conversions done ------")
