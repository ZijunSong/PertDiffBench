import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_umap_comparison_plot():
    """
    Reads training, test, and four predicted scRNA-seq datasets,
    and generates a 1x4 UMAP comparison plot with detailed cell categories.
    """
    # --- 1. Configuration: Update file paths here ---
    # This dictionary holds the paths to your 6 data files.
    # The keys ('train', 'test', 'pred1', etc.) are used as labels.
    file_paths = {
        'train': 'data/fig1/task1/task1_train_CD4T_exp.h5ad',
        'test': 'data/fig1/task1/task1_valid_B_exp.h5ad',
        'pred1': 'samples/fig2/task2_unseen_celltype/B/squidiff_1000/synthetic_ifn_run_1.h5ad', # Squdiff
        'pred2': 'samples/fig2/task2_unseen_celltype/sample_B/scDiffusion/synthetic_ifn_1.h5ad', # scDiffusion
        'pred3': 'samples/fig2/task2_unseen_celltype/pretrain_CD4T_B/scrna_ddpm_scrna/synthetic_ifn_1.h5ad', # DDPM
        'pred4': 'samples/fig2/task2_unseen_celltype/pretrain_CD4T_B/mlp_ddpm_mlp/synthetic_ifn_1.h5ad', # DDPM+MLP
    }
    
    # Output file for the final plot
    output_plot_path = "figs/fig2/umap_comparison_1x4_B.svg"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    # --- 2. Load all AnnData objects ---
    print("--- Loading data files ---")
    try:
        adata_objects = {key: sc.read_h5ad(path) for key, path in file_paths.items()}
        adata_train = adata_objects.pop('train')
        adata_test = adata_objects.pop('test')
        # The rest of adata_objects are the prediction datasets
        adata_predictions = adata_objects
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}.")
        print("Please ensure the file_paths in the script are correct.")
        return

    print("✔ Data loaded successfully.")

    # --- 3. Create a fixed reference UMAP embedding ---
    print("\n--- Creating reference UMAP layout ---")
    # Add a source label to each dataset before combining
    adata_train.obs['data_source'] = 'train'
    adata_test.obs['data_source'] = 'test'
    
    # Ensure observation names are unique before concatenation
    adata_train.obs_names_make_unique()
    adata_test.obs_names_make_unique()

    # Combine train and test data to form the reference background
    adata_ref = sc.concat([adata_train, adata_test], join='outer', index_unique=None, fill_value=0)
    
    # Calculate neighbors and the UMAP embedding on the reference data
    # This creates the fixed layout that everything will be projected onto
    print("Calculating neighbors and UMAP for reference data...")
    sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep='X', random_state=0)
    sc.tl.umap(adata_ref, random_state=0)
    print("✔ Reference UMAP created.")
    
    # The 'ingest' function requires the .raw attribute to be set
    adata_ref.raw = adata_ref

    # --- 4. Setup Matplotlib Figure ---
    # Create a figure with 1 row and 4 columns of subplots.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes = axes.flatten()
    
    prediction_titles = {
        'pred1': 'Squdiff',
        'pred2': 'scDiffusion',
        'pred3': 'DDPM',
        'pred4': 'DDPM+MLP',
    }

    # --- 5. Loop, Project, and Plot for each prediction file ---
    # Define the desired order of plots by their keys from the file_paths dictionary
    plot_order = ['pred3', 'pred4', 'pred2', 'pred1'] # DDPM, DDPM+MLP, scDiffusion, Squdiff
    
    print("\n--- Projecting and plotting prediction datasets ---")
    for i, key in enumerate(plot_order):
        adata_pred = adata_predictions[key]
        ax = axes[i]
        print(f"Processing: {key} ({prediction_titles.get(key, key.capitalize())})...")

        # a) Project the synthetic data onto the reference UMAP
        adata_pred.obs['data_source'] = 'generated'
        adata_pred.obs_names_make_unique()
        
        sc.tl.ingest(adata_pred, adata_ref, embedding_method='umap')

        # b) Combine the reference and projected data for visualization
        adata_viz = sc.concat([adata_ref, adata_pred], join='inner', index_unique=None)
        
        # c) Create a categorical column for plotting layers with more detail
        # Initialize with a default value
        adata_viz.obs['plot_group'] = 'Other'
        
        # Assign categories based on data source and perturbation status
        is_train = adata_viz.obs['data_source'] == 'train'
        adata_viz.obs.loc[is_train, 'plot_group'] = 'Training Cells (CD4T)'
        
        is_control_test = (adata_viz.obs['data_source'] == 'test') & (adata_viz.obs['perturbation_status'] == 'Control')
        adata_viz.obs.loc[is_control_test, 'plot_group'] = 'Control Cells (B)'
        
        is_true_pert = (adata_viz.obs['data_source'] == 'test') & (adata_viz.obs['perturbation_status'] != 'Control')
        adata_viz.obs.loc[is_true_pert, 'plot_group'] = 'True Perturbed (B)'
        
        is_generated = adata_viz.obs['data_source'] == 'generated'
        adata_viz.obs.loc[is_generated, 'plot_group'] = 'Generated Perturbed (B)'

        # d) Define the plotting order and colors for consistency
        category_order = [
            'Training Cells (CD4T)',
            'Control Cells (B)',
            'True Perturbed (B)', 
            'Generated Perturbed (B)'
        ]
        color_palette = {
            'Training Cells (CD4T)': 'lightgray',
            'Control Cells (B)': 'dimgray',
            'True Perturbed (B)': 'blue',
            'Generated Perturbed (B)': 'orange'
        }
        
        adata_viz.obs['plot_group'] = pd.Categorical(adata_viz.obs['plot_group'], categories=category_order)

        # e) Plot using sc.pl.umap on the designated subplot axis
        sc.pl.umap(
            adata_viz,
            color='plot_group',
            palette=color_palette,
            ax=ax,
            size=10, # Adjust point size as needed
            title=prediction_titles.get(key, key.capitalize()),
            legend_loc='best', # Use a standard legend box
            legend_fontsize=8,
            show=False # Do not show the plot immediately
        )
        
    # --- 6. Finalize and Save the Figure ---
    print("\n--- Finalizing plot ---")
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()
    
    # Save the complete figure to a file
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"✔ Successfully saved UMAP comparison plot to: {output_plot_path}")
    plt.close(fig) # Close the figure to free up memory

if __name__ == "__main__":
    generate_umap_comparison_plot()
