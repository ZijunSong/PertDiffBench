import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

# Directory containing the input CSV files
data_dir = '/data/ppnm/data/PertDiffBench/data_ori/fig1/raw_task1/'  # TODO: Change to your data path

print(f"Processing CSV files in directory: {data_dir}...")

# Collect all CSV files
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"Error: No CSV files found in {data_dir}.")
    exit()

for csv_file in csv_files:
    # Derive H5AD filename from the CSV filename
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(data_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- Processing file: {csv_file} ---")

    # Read CSV
    try:
        df = pd.read_csv(csv_file, index_col=0)
        print("CSV loaded successfully.")
        print(f"Data shape (cells x genes): {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}. Please check the path.")
        continue
    except Exception as e:
        print(f"Error reading CSV {csv_file}: {e}")
        continue

    # Create AnnData object
    adata = ad.AnnData(df)
    print("AnnData object created.")
    # print(adata)  # Uncomment to inspect details

    # Add metadata to adata.obs
    print("Adding metadata to adata.obs ...")

    # Extract 'Cell.Type' from filename
    try:
        cell_type = base_name.split('_')[2]
        adata.obs['Cell.Type'] = cell_type
        print(f"Added 'Cell.Type' column: {cell_type}.")
    except IndexError:
        print("Warning: Could not parse 'Cell.Type' from filename. Using 'unknown'.")
        adata.obs['Cell.Type'] = 'unknown'

    # Add 'perturbation_status' based on cell ID suffix
    conditions = [
        adata.obs.index.str.endswith('stimulated'),
        adata.obs.index.str.endswith('control')
    ]
    choices = ['IFN', 'Control']

    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')
    print("Added 'perturbation_status' column.")

    # Metadata update complete
    print("Metadata added.")
    # print(adata)  # Uncomment to inspect details

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(adata.X)

    # Write AnnData to H5AD file
    try:
        adata.write(h5ad_file)
        print(f"Saved (with metadata) to: {h5ad_file}")
    except Exception as e:
        print(f"Error saving H5AD file: {e}")

    # Optional: read back and verify (skip for many files to save time)
    print("--- Verification ---")
    print(f"Reading back saved H5AD: {h5ad_file}")
    try:
        adata_loaded = ad.read_h5ad(h5ad_file)
        print("File loaded successfully. Loaded object info:")
        # print(adata_loaded)  # Uncomment to inspect details
        print("Loaded metadata (first 5 rows):")
        print(adata_loaded.obs.head())
    except Exception as e:
        print(f"Error reading back H5AD {h5ad_file}: {e}")

print("\n------ All files converted ------")