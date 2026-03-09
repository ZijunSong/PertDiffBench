import pandas as pd
import anndata as ad
import numpy as np
import os
import glob
import scanpy as sc

data_dir = 'data/fig1/raw_task3/'
output_dir = 'data/fig1/hvg_task3/'
os.makedirs(output_dir, exist_ok=True)

print(f"Starting to process H5AD files in directory: {data_dir}...")

h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))

if not h5ad_files:
    print(f"Error: No H5AD files found in {data_dir}. Please ensure you have run the previous scripts to generate the H5AD files.")
    exit()

target_cell_types = [f'mix{i}' for i in range(2, 8)]
target_dataset_types = ['train', 'test']
hvg_numbers = [1000]

for target_cell_type in target_cell_types:
    print(f"\n==================== Starting to process cell type: {target_cell_type} ====================")
    for current_dataset_type in target_dataset_types:
        print(f"\n--- Processing '{current_dataset_type}' dataset for '{target_cell_type}' cell type ---")

        selected_h5ad_file = None
        for h5ad_file in h5ad_files:
            base_name = os.path.basename(h5ad_file).lower()
            if target_cell_type.lower() in base_name and current_dataset_type.lower() in base_name:
                selected_h5ad_file = h5ad_file
                break

        if selected_h5ad_file is None:
            print(f"Error: No H5AD file found for '{target_cell_type}' and '{current_dataset_type}'. Skipping this dataset.")
            continue

        print(f"Selected file: {os.path.basename(selected_h5ad_file)}")

        try:
            adata_selected_cell_type = ad.read_h5ad(selected_h5ad_file)
            print("AnnData file read successfully.")
            print(f"Data dimensions (cells x genes): {adata_selected_cell_type.shape}")
            if 'Cell.Type' in adata_selected_cell_type.obs and adata_selected_cell_type.obs['Cell.Type'].iloc[0] != target_cell_type:
                print(f"Warning: The actual cell type in the loaded file '{adata_selected_cell_type.obs['Cell.Type'].iloc[0]}' does not match the target '{target_cell_type}'. Please check the file.")

        except Exception as e:
            print(f"Error reading H5AD file {selected_h5ad_file}: {e}")
            continue

        adata_hvg_processed = adata_selected_cell_type.copy()

        print("Performing data normalization and log transformation...")
        sc.pp.normalize_total(adata_hvg_processed, target_sum=1e4)
        sc.pp.log1p(adata_hvg_processed)
        print("Normalization and log transformation completed.")

        print(f"--- Generating files with different numbers of highly variable genes for '{current_dataset_type}' dataset ---")

        num_genes_in_data = adata_hvg_processed.shape[1]
        n_top_genes_to_calculate = min(max(hvg_numbers), num_genes_in_data)

        print(f"Calculating highly variable genes, target number: {n_top_genes_to_calculate} (or all genes if fewer exist)...")
        sc.pp.highly_variable_genes(adata_hvg_processed, n_top_genes=n_top_genes_to_calculate, flavor='seurat_v3', subset=False)
        print("Highly variable gene calculation completed.")

        for n_hvg in hvg_numbers:
            if n_hvg > num_genes_in_data:
                print(f"Warning: The requested number of highly variable genes {n_hvg} is greater than the total number of genes {num_genes_in_data}. All genes will be used.")
                current_n_hvg = num_genes_in_data
            else:
                current_n_hvg = n_hvg

            top_hvg_genes_indices = adata_hvg_processed.var.sort_values('highly_variable_rank').index[:current_n_hvg]
            
            adata_filtered_hvg = adata_selected_cell_type[:, top_hvg_genes_indices].copy()
            
            output_filename = os.path.join(output_dir, f"{target_cell_type}_{current_dataset_type}_HVG_{n_hvg}.h5ad")

            try:
                adata_filtered_hvg.write(output_filename)
                print(f"Successfully saved file: {output_filename} (containing {adata_filtered_hvg.shape[1]} highly variable genes)")
            except Exception as e:
                print(f"Error saving file {output_filename}: {e}")

print("\n------ All highly variable gene files have been generated ------")
