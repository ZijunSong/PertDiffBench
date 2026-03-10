import pandas as pd
import anndata as ad
import numpy as np
import os
import glob
import scanpy as sc

# --- Configuration ---
data_dir = '/data/ppnm/data/PertDiffBench/data_ori/fig1/raw_task1/'  # TODO: Change to your data path
output_dir = '/data/ppnm/data/PertDiffBench/data/highly_variable_gene_gradient'  # TODO: Change to your data path
os.makedirs(output_dir, exist_ok=True)

# Define target cell types and dataset types to process
target_cell_types = ['NK', 'CD4T', 'CD8T', 'B', 'CD14+Mono', 'Dendritic', 'FCGR3A+Mono'] 
target_dataset_types = ['train', 'valid'] 

# Define the gradient of highly variable gene counts to generate
hvg_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 6998]

# --- Main Processing Logic ---
print(f"Starting to process H5AD files in directory: {data_dir}...")

# Get all H5AD files from the input directory
h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))

if not h5ad_files:
    print(f"Error: No H5AD files found in {data_dir}. Please ensure you have run the previous scripts to generate the H5AD files.")
    exit()

# Iterate over each target cell type
for target_cell_type in target_cell_types:
    print(f"\n==================== Processing Cell Type: {target_cell_type} ====================")
    
    # Iterate over each target dataset type (e.g., 'train', 'valid')
    for current_dataset_type in target_dataset_types:
        print(f"\n--- Processing '{current_dataset_type}' dataset for '{target_cell_type}' cell type ---")

        # Find the correct H5AD file for the current loop iteration
        selected_h5ad_file = None
        for h5ad_file in h5ad_files:
            base_name = os.path.basename(h5ad_file).lower() # Convert to lowercase for robust matching
            # Find a file containing both the target cell type and the current dataset type
            if target_cell_type.lower() in base_name and current_dataset_type.lower() in base_name:
                selected_h5ad_file = h5ad_file
                break # Exit after finding the first matching file

        if selected_h5ad_file is None:
            print(f"Error: No H5AD file found matching '{target_cell_type}' and '{current_dataset_type}'. Skipping this dataset.")
            continue # Skip the current dataset and proceed to the next

        print(f"Selected file: {os.path.basename(selected_h5ad_file)}")

        # Load the AnnData file
        try:
            adata_selected_cell_type = ad.read_h5ad(selected_h5ad_file)
            print("AnnData file read successfully.")
            print(f"Data dimensions (cells x genes): {adata_selected_cell_type.shape}")
            
            # Verify if the cell type is consistent with the target (optional but recommended)
            if 'Cell.Type' in adata_selected_cell_type.obs and adata_selected_cell_type.obs['Cell.Type'].iloc[0] != target_cell_type:
                print(f"Warning: The actual cell type '{adata_selected_cell_type.obs['Cell.Type'].iloc[0]}' in the loaded file does not match the target '{target_cell_type}'. Please check the file.")

        except Exception as e:
            print(f"Error reading H5AD file {selected_h5ad_file}: {e}")
            continue

        # --- Preprocessing for HVG Selection ---
        adata_hvg_processed = adata_selected_cell_type.copy()

        print("Performing data normalization and log transformation for HVG selection...")
        # Normalize total counts per cell to 10,000
        sc.pp.normalize_total(adata_hvg_processed, target_sum=1e4)
        # Apply log1p transformation to the data
        sc.pp.log1p(adata_hvg_processed)
        print("Normalization and log transformation complete.")

        print(f"--- Generating files with different numbers of highly variable genes for '{current_dataset_type}' dataset ---")

        # Calculate highly variable genes once for the maximum required number.
        num_genes_in_data = adata_hvg_processed.shape[1]
        n_top_genes_to_calculate = min(max(hvg_numbers), num_genes_in_data)

        print(f"Calculating highly variable genes, targeting: {n_top_genes_to_calculate} (or all genes if fewer exist)...")
        # Calculate HVGs using 'seurat_v3' method and annotate genes in .var, without subsetting the data yet.
        sc.pp.highly_variable_genes(adata_hvg_processed, n_top_genes=n_top_genes_to_calculate, flavor='seurat_v3', subset=False)
        print("Highly variable gene calculation complete.")

        # --- Generate and Save Subsets ---
        # Generate and save files for each number of HVGs in the gradient
        for n_hvg in hvg_numbers:
            current_n_hvg = n_hvg
            if n_hvg > num_genes_in_data:
                print(f"Warning: Requested number of HVGs {n_hvg} is greater than the total number of genes {num_genes_in_data}. Using all {num_genes_in_data} genes instead.")
                current_n_hvg = num_genes_in_data

            # Sort by 'highly_variable_rank' and select the indices of the top 'current_n_hvg' genes
            top_hvg_gene_names = adata_hvg_processed.var.sort_values('highly_variable_rank').index[:current_n_hvg]

            # Select these highly variable genes from the *original* (unnormalized) data
            # This ensures the output file contains the original count data for flexible downstream analysis
            adata_filtered_hvg = adata_selected_cell_type[:, top_hvg_gene_names].copy()

            # Construct the output filename
            output_filename = os.path.join(output_dir, f"{target_cell_type}_{current_dataset_type}_HVG_{n_hvg}.h5ad")

            try:
                adata_filtered_hvg.write(output_filename)
                print(f"Successfully saved file: {output_filename} (containing {adata_filtered_hvg.shape[1]} highly variable genes)")
            except Exception as e:
                print(f"Error saving file {output_filename}: {e}")

print("\n------ All highly variable gene files have been generated ------")

