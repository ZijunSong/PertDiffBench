import argparse
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import scgen

from utils.metrics import (
    compute_mae,
    compute_des,
    compute_pds,
    compute_edistance,
    compute_r2,
    compute_mmd,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_delta_de,
)

def get_de_genes_from_anndatas(adata_ctrl, adata_pert, group_name='group', ctrl_name='Control', pert_name='Perturbed', pval_cutoff=0.05):
    """
    Helper function to find Differentially Expressed Genes by comparing two AnnData objects.
    """
    # Combine control and perturbed data for comparison
    combined_adata = adata_ctrl.concatenate(adata_pert, batch_key=group_name, batch_categories=[ctrl_name, pert_name])
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(combined_adata, groupby=group_name, method='wilcoxon', groups=[pert_name], reference=ctrl_name)
    
    result = combined_adata.uns['rank_genes_groups']
    
    # Filter for significant genes
    de_mask = result['pvals_adj'][pert_name] < pval_cutoff
    de_genes = set(np.array(result['names'][pert_name])[de_mask])
    
    # Create a dictionary of all ranked genes and their log fold changes for the DES function
    all_genes = result['names'][pert_name]
    all_lfcs = result['logfoldchanges'][pert_name]
    gene_fold_changes = dict(zip(all_genes, all_lfcs))
    
    return de_genes, gene_fold_changes

parser = argparse.ArgumentParser(description='Predict perturbation and evaluate using scGen.')
parser.add_argument('--train_data_path', type=str, default='data/scrna_data/scrna_positive.h5ad', help='Path to the training set .h5ad file')
parser.add_argument('--test_data_path', type=str, default=None, help='Path to the test set .h5ad file (optional)')
parser.add_argument('--model_save_path', type=str, default='checkpoints/scgen/model_perturbation_prediction', help='Path to save/load the model')
parser.add_argument('--celltype_to_predict', type=str, default='B', help='Cell type to predict')
parser.add_argument('--out_h5ad', type=str, default='results/scgen/synthetic_prediction.h5ad', help='Path to save the final synthetic .h5ad results')
parser.add_argument('--umap_plot', type=str, default="results/scgen/umap_comparison.png")
parser.add_argument('--n_samples', type=int, default=100, help='Number of cells to sample for evaluation')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True)

# 1. Load training set (for model initialization)
print("="*20, "1. Loading training set", "="*20)
try:
    train_adata = sc.read(args.train_data_path)
    print("Training set loaded successfully:", train_adata)
except FileNotFoundError:
    print(f"Error: Training set file not found at '{args.train_data_path}'")
    exit()

# 2. Load test set (if not provided, defaults to the training set)
test_adata = None
if args.test_data_path and os.path.exists(args.test_data_path):
    print("="*20, "2. Loading test set", "="*20)
    test_adata = sc.read(args.test_data_path)
    print("Test set loaded successfully:", test_adata)
else:
    print("Test set not provided, will use the training set as the test set.")
    test_adata = train_adata.copy()

# 3. Initialize the model (must use the training set)
scgen.SCGEN.setup_anndata(
    train_adata, 
    batch_key="perturbation_status", 
    labels_key="Cell.Type"
)
model = scgen.SCGEN(train_adata)
print("Model initialized.")

# 4. Check for a pre-trained model or train a new one
if os.path.exists(os.path.join(args.model_save_path, "model.pt")):
    print("\n" + "="*20, "4. Loading pre-trained model", "="*20)
    model = scgen.SCGEN.load(args.model_save_path, adata=train_adata)
    print("Pre-trained model loaded.")
else:
    print("\n" + "="*20, "4. Training a new model", "="*20)
    model.train(
        max_epochs=100,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=25,
        check_val_every_n_epoch=1,
    )
    model.save(args.model_save_path, overwrite=True)
    print(f"Model saved to: {args.model_save_path}")

# 5. Predict on the test set
print("\n" + "="*20, "5. Predicting perturbation on the test set", "="*20)
model.adata = test_adata  # Crucial: Replace the model's internal data with the test set
pred_adata, delta = model.predict(
    ctrl_key='Control',
    stim_key='IFN',
    celltype_to_predict=args.celltype_to_predict
)
pred_adata.obs['perturbation_status'] = f"pred_{args.celltype_to_predict}"

# 6. Calculate evaluation metrics
print("\n" + "="*50)
print(f"   Evaluation Metrics for {args.celltype_to_predict} (predicted)")
print("="*50)

# Get true control and stimulated cells for the target cell type
ctrl_mask = (model.adata.obs['perturbation_status'] == "Control") & (model.adata.obs['Cell.Type'] == args.celltype_to_predict)
stim_mask = (model.adata.obs['perturbation_status'] == "IFN") & (model.adata.obs['Cell.Type'] == args.celltype_to_predict)

ctrl_adata_true = model.adata[ctrl_mask].copy()
stim_adata_true = model.adata[stim_mask].copy()

if ctrl_adata_true.n_obs < args.n_samples or stim_adata_true.n_obs < args.n_samples:
    print(f"Error: Not enough cells for evaluation. Need at least {args.n_samples} for control and stimulated groups.")
    sys.exit(1)

# Sample cells for fair comparison
sc.pp.subsample(ctrl_adata_true, n_obs=args.n_samples, random_state=0)
sc.pp.subsample(stim_adata_true, n_obs=args.n_samples, random_state=0)
sc.pp.subsample(pred_adata, n_obs=args.n_samples, random_state=0)

# Get expression matrices
ctrl_X = ctrl_adata_true.X.toarray() if hasattr(ctrl_adata_true.X, 'toarray') else ctrl_adata_true.X
true_pert_X = stim_adata_true.X.toarray() if hasattr(stim_adata_true.X, 'toarray') else stim_adata_true.X
pred_X = pred_adata.X.toarray() if hasattr(pred_adata.X, 'toarray') else pred_adata.X

# Create pseudobulk profiles
true_pert_pb = np.mean(true_pert_X, axis=0)
pred_pert_pb = np.mean(pred_X, axis=0)
ctrl_pb = np.mean(ctrl_X, axis=0)

# --- Calculate DES ---
print("Calculating Differentially Expressed Genes (DEGs)...")
# 1. Find TRUE DEGs by comparing true stimulated vs. true control cells
true_de_genes, _ = get_de_genes_from_anndatas(ctrl_adata_true, stim_adata_true, pert_name='True_Stim')
# 2. Find PREDICTED DEGs by comparing predicted stimulated vs. true control cells
pred_de_genes, pred_gene_fold_changes = get_de_genes_from_anndatas(ctrl_adata_true, pred_adata, pert_name='Pred_Stim')
print(f"Found {len(true_de_genes)} true DEGs and {len(pred_de_genes)} predicted DEGs.")

# Calculate all other metrics
mae_val = compute_mae(true_pert_pb, pred_pert_pb)
des_val = compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes)
pds_val = compute_pds(pred_pert_pb.reshape(1,-1), true_pert_pb.reshape(1,-1)) # Will be 1.0 for single pert
edist_val = compute_edistance(true_pert_X, pred_X)
mmd_val = compute_mmd(true_pert_X, pred_X)
r2_val = compute_r2(true_pert_X, pred_X)
p_all = compute_pearson(true_pert_pb, pred_pert_pb)
pd_all = compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb)
pd_de20 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=20)
pd_de50 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=50)
pd_de100 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=100)

# Print results
print(f"Perturbation Discrimination Score (PDS): {pds_val:.4f} (Note: 1.0 is expected for single perturbation eval)")
print(f"Mean Absolute Error (MAE): {mae_val:.4f}")
print(f"Differential Expression Score (DES): {des_val:.4f}") 
print("-" * 20)
print(f"E-Distance: {edist_val:.4f}")
print(f"Maximum Mean Discrepancy (MMD): {mmd_val:.4f}")
print(f"R-squared (R2): {r2_val:.4f}")
print("-" * 20)
print(f"Pearson (all genes): {p_all:.4f}")
print(f"Pearson Delta (all genes): {pd_all:.4f}")
print(f"Pearson Delta (top 20 DE genes): {pd_de20:.4f}")
print(f"Pearson Delta (top 50 DE genes): {pd_de50:.4f}")
print(f"Pearson Delta (top 100 DE genes): {pd_de100:.4f}")
print("="*50 + "\nEvaluation complete!")

# --- 5. UMAP Visualization ---
print("\n--- Generating UMAP Visualization ---")
# Combine true control, true stimulated, and predicted cells
adata_viz = sc.concat([ctrl_adata_true, stim_adata_true, pred_adata], join='inner', index_unique=None)

# Standard preprocessing for visualization
sc.pp.neighbors(adata_viz, n_neighbors=15, use_rep='X')
sc.tl.umap(adata_viz)

# Plot UMAP
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata_viz, color='perturbation_status', ax=ax, show=False, title=f"UMAP of True and Predicted Cells ({args.celltype_to_predict})")

# Save the plot
os.makedirs(os.path.dirname(args.umap_plot), exist_ok=True)
plt.savefig(args.umap_plot, dpi=300, bbox_inches='tight')
print(f"✔ Saved UMAP plot to {args.umap_plot}")
plt.close(fig)

# --- 6. Save Output ---
pred_adata.write_h5ad(args.out_h5ad)
print(f"✔ Saved synthetic AnnData to {args.out_h5ad}")