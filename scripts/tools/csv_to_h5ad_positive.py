"""
Convert expression & metadata CSVs into an AnnData .h5ad file, print detailed CSV information,
and output detailed AnnData information.
This version includes a 'perturbation_status' column in adata.obs for easier distinction
between control and treated samples.
"""
import argparse
import pandas as pd
import anndata as ad
import scanpy as sc

def main(args):
    # Load expression matrix (genes x cells)
    exp_df = pd.read_csv(args.exp_csv, index_col=0)
    # Load metadata (cells x annotations)
    meta_df = pd.read_csv(args.meta_csv, index_col=0)

    # Print the first few rows of both CSVs for preview
    print("\n--- Expression Data Preview (first 5 rows) ---")
    print(exp_df.head())
    
    print("\n--- Metadata Preview (first 5 rows) ---")
    print(meta_df.head())

    # Check for common cells between the expression and metadata
    common_cells = exp_df.columns.intersection(meta_df.index)
    print(f"\nFound {len(common_cells)} common cells between expression and metadata.")
    
    # Keep only common cells in both tables
    # Ensure consistent ordering if necessary, though intersection doesn't guarantee order.
    # For AnnData, obs_names (from meta_df.index) should align with X's first dimension (cells).
    # The expression matrix is genes x cells, so after transpose it will be cells x genes.
    exp_df  = exp_df[common_cells]
    meta_df = meta_df.loc[common_cells]

    # Build AnnData: observations (cells) x variables (genes)
    # The expression matrix exp_df is currently genes (rows) x cells (columns).
    # AnnData expects X as cells (rows) x genes (columns). So, transpose exp_df.
    adata = ad.AnnData(
        X=exp_df.T.values,  # Transpose for cells x genes
        obs=meta_df,        # Cell metadata, index should match cells in X
        var=pd.DataFrame(index=exp_df.index) # Gene metadata, index should match genes in X
    )
    # Add gene names to 'var' (gene metadata)
    adata.var['gene_names'] = adata.var_names # or exp_df.index.tolist()

    # --- Filtering for Highly Variable Genes ---
    print("\n--- Filtering for highly variable genes ---")
    if adata.n_vars > 0 : # Ensure there are genes to filter
        # Store raw counts in .layers['counts'] for HVG calculation as specified by layer="counts"
        # adata.X currently holds the raw counts (or whatever was in exp_df.T)
        adata.layers["counts"] = adata.X.copy()

        try:
            # Note: flavor="seurat_v3" typically expects log-normalized data,
            # but using layer="counts" will apply the method to the raw counts in that layer.
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=2000,
                subset=True, # Subsets adata to only include highly variable genes
                layer="counts",
                flavor="seurat_v3"
            )
            print(f"Filtered AnnData to {adata.n_vars} highly variable genes (requested top 2000).")
            # After subsetting, adata.var['gene_names'] will also be correctly subsetted.
        except Exception as e:
            print(f"Error during highly variable gene selection: {e}")
            print("Skipping HVG selection or HVG selection resulted in no change.")
            # If subset=True and it fails or selects all genes, adata.n_vars would reflect that.
    else:
        print("No genes found in AnnData object. Skipping HVG selection.")
        
    # --- New feature: Add perturbation_status column for easier differentiation ---
    # Based on the 'Condition' column:
    # - If 'Condition' starts with 'Control', assign 'Control'.
    # - If 'Condition' starts with 'IFN', assign 'IFN'.
    # This makes downstream analysis and visualization of perturbation effects simpler.
    def assign_perturbation_status(condition_value):
        if isinstance(condition_value, str):
            if condition_value.startswith('Control'):
                return 'Control'
            elif condition_value.startswith('IFN'):
                return 'IFN'
        return 'Unknown' # Default for unexpected values

    if 'Condition' in adata.obs.columns:
        adata.obs['perturbation_status'] = adata.obs['Condition'].apply(assign_perturbation_status)
        print("\nAdded 'perturbation_status' column to AnnData observation metadata (adata.obs).")
    else:
        print("\nWarning: 'Condition' column not found in metadata. Cannot create 'perturbation_status'.")


    # Write to .h5ad
    adata.write_h5ad(args.out_h5ad)
    print(f"Saved AnnData to {args.out_h5ad}")

    # Print detailed AnnData information
    print("\n--- AnnData Information ---")
    print(adata) # AnnData object now has a concise print format
    print(f"AnnData shape (cells x genes): {adata.shape}")
    print(f"Number of cells (observations): {adata.n_obs}")
    print(f"Number of genes (variables): {adata.n_vars}")
    print(f"Observation (cell metadata) columns: {adata.obs.columns.tolist()}")
    print(f"Variable (gene metadata) columns: {adata.var.columns.tolist()}")
    
    # Preview of expression matrix from AnnData (first 5 cells/genes)
    print("\n--- Expression Data from AnnData (first 5x5 slice) ---")
    # Ensure there are at least 5 cells and 5 genes before slicing
    if adata.n_obs >= 5 and adata.n_vars >= 5:
        print(adata.X[:5, :5])
    elif adata.n_obs > 0 and adata.n_vars > 0:
        print(adata.X[:min(5, adata.n_obs), :min(5, adata.n_vars)]) # Print available slice
    else:
        print("Expression data is empty or too small to preview 5x5 slice.")


    # Print the first 5 rows of observation metadata (obs)
    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    # Print the first 5 rows of gene metadata (var)
    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert CSV expression and metadata to H5AD format. Adds a 'perturbation_status' column for clarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    p.add_argument("--exp_csv", type=str, default="data/scrna_data/NK_IFN_exp_count_allgenes4scvi.csv",
                   help="Path to gene-expression CSV (genes x cells format)")
    p.add_argument("--meta_csv", type=str, default="data/scrna_data/NK_IFN_meta.csv",
                   help="Path to cell-metadata CSV (cells x annotations format)")
    p.add_argument("--out_h5ad", type=str, default="data/scrna_data/scrna_positive.h5ad",
                   help="Output .h5ad filepath")
    args = p.parse_args()
    main(args)