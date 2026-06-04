import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

# Directory containing input CSV files
data_dir = 'data/fig1/task3/'

print(f"Processing CSV files in: {data_dir}...")

csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"Error: no CSV files found in {data_dir}.")
    exit()

for csv_file in csv_files:
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(data_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- Processing: {csv_file} ---")

    try:
        # Skip the first column (row id); use the second column as the cell index.
        df = pd.read_csv(csv_file, index_col=1)
        print("CSV loaded successfully.")
        print(f"Shape (cells x genes): {df.shape}")
    except FileNotFoundError:
        print(f"Error: file not found: {csv_file}")
        continue
    except Exception as e:
        print(f"Error reading CSV {csv_file}: {e}")
        continue

    # Data are already cells x genes; no transpose needed.
    df_expression = df
    print("Confirmed cells x genes format; no transpose needed.")

    adata = ad.AnnData(df_expression)
    print("AnnData object created.")
    # print(adata)  # Uncomment for verbose details if needed

    print("Adding metadata to adata.obs ...")

    try:
        cell_type = base_name.split('_')[2]
        adata.obs['Cell.Type'] = cell_type
        print(f"Added Cell.Type column: {cell_type}.")
    except IndexError:
        print("Warning: could not parse Cell.Type from filename; using 'unknown'.")
        adata.obs['Cell.Type'] = 'unknown'

    conditions = [
        adata.obs.index.str.endswith('control'),
        adata.obs.index.str.endswith('stimulated'),
    ]
    choices = ['Control', 'IFN']

    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')
    print("Added perturbation_status from cell ID suffix.")

    print("Metadata added.")
    # print(adata)  # Uncomment for verbose details if needed

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(adata.X)

    try:
        adata.write(h5ad_file)
        print(f"Saved H5AD with metadata: {h5ad_file}")
    except Exception as e:
        print(f"Error saving H5AD: {e}")

    print("--- Validation ---")
    print(f"Reloading: {h5ad_file}")
    try:
        adata_loaded = ad.read_h5ad(h5ad_file)
        print("Reload OK. Loaded object:")
        # print(adata_loaded)  # Uncomment for verbose details if needed
        print("obs head after reload:")
        print(adata_loaded.obs.head())
        print("var head after reload:")
        print(adata_loaded.var.head())
    except Exception as e:
        print(f"Error reloading H5AD {h5ad_file}: {e}")

print("\n------ All conversions finished ------")
