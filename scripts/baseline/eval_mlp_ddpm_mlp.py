# scripts/eval_mlp_ddpm_mlp.py

import os
import sys
import argparse
from collections import defaultdict

from omegaconf import OmegaConf
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import required evaluation metrics
from utils.metrics import (
    compute_mae,
    compute_des,
    compute_pds,
    compute_overall_score, # Note: You'll need baseline scores to use this
    compute_edistance,
    compute_r2,
    compute_mmd,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_de,
    compute_pearson_delta_de,
)
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP

def main():
    # Set random seeds for reproducibility across all relevant libraries
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        # The following two lines are for fully reproducible results on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Evaluate MLP-DDPM-MLP baseline on scRNA-seq: generate perturbed from control and compute metrics"
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/mlp_ddpm_mlp.yaml",
        help="Path to the OmegaConf config"
    )
    parser.add_argument(
        "-k", "--ckpt",
        default="./checkpoints/mlp_ddpm_mlp/ckpt_100.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "-n", "--n_samples",
        type=int,
        default=100,
        help="Number of control cells to generate and evaluate per perturbation"
    )
    parser.add_argument(
        "-o", "--out_h5ad",
        default="samples/mlp_ddpm_mlp.h5ad",
        help="Output path for synthetic AnnData"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data.path in the YAML file (this is the test set)"
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to the training AnnData file (.h5ad) for UMAP visualization"
    )
    parser.add_argument(
        "--gene-nums",
        type=int,
        default=None,
        help="Override model.input_dim in the YAML file"
    )
    parser.add_argument(
        "--umap_plot",
        type=str,
        default="samples/umap_comparison.png",
        help="Output path for the UMAP comparison plot"
    )
    args = parser.parse_args()

    # 1) Load config and model
    cfg = OmegaConf.load(args.config)

    if args.gene_nums:
        print(f"Overriding model.input_dim from command line: '{cfg.model.ae.input_dim}' -> '{args.gene_nums}'")
        cfg.model.ae.input_dim = args.gene_nums

    if args.data_path:
        print(f"Overriding data.path from command line: '{cfg.data.path}' -> '{args.data_path}'")
        cfg.data.path = args.data_path

    if args.out_h5ad:
        print(f"Overriding out_h5ad from command line: '{cfg.sample.out_h5ad}' -> '{args.out_h5ad}'")
        cfg.sample.out_h5ad = args.out_h5ad

    device = torch.device(cfg.train.device)
    model = MLPDDPMMLP(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✔ Loaded checkpoint from {args.ckpt}")

    # 2) Read raw data (test set) and identify perturbations
    adata = sc.read_h5ad(cfg.data.path)
    ctrl_mask = adata.obs["perturbation_status"] == "Control"
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()
    
    perturbations = adata.obs["perturbation_status"].unique().tolist()
    perturbations = [p for p in perturbations if p != "Control"]
    print(f"Found {len(perturbations)} perturbations to evaluate in test set: {perturbations}")

    # 3) Pre-evaluation check for n_samples
    print("\n--- Checking sample counts ---")
    ctrl_count = len(ctrl_ids)
    pert_counts = {p: np.sum(adata.obs["perturbation_status"] == p) for p in perturbations}
    
    if not pert_counts:
        print("Warning: No perturbation groups found in the data. Exiting.")
        return

    min_pert_count = min(pert_counts.values())
    max_possible_samples = min(ctrl_count, min_pert_count)

    print(f"Control cells available in test set: {ctrl_count}")
    print(f"Minimum cells in a perturbation group in test set: {min_pert_count}")
    print(f"Maximum possible --n_samples: {max_possible_samples}")

    if args.n_samples > max_possible_samples:
        print(f"\nError: --n_samples ({args.n_samples}) exceeds the maximum possible value ({max_possible_samples}).")
        print("This value is limited by the smallest perturbation group or the control group in the test set.")
        print("Please specify a value for --n_samples that is less than or equal to this maximum.")
        sys.exit(1) # Exit the script

    all_pred_pb = []
    all_true_pb = []
    all_ctrl_pb = []
    metrics_results = defaultdict(list)
    all_synthetic_adata = []

    # 4) Loop through each perturbation to generate predictions and calculate metrics
    for pert in perturbations:
        print(f"\n--- Evaluating perturbation: {pert} ---")
        pert_mask = adata.obs["perturbation_status"] == pert
        pert_ids = adata.obs_names[pert_mask].tolist()

        selected_ctrl_ids = np.random.choice(ctrl_ids, args.n_samples, replace=False)
        selected_pert_ids = np.random.choice(pert_ids, args.n_samples, replace=False)
        
        ctrl_X_data = adata[selected_ctrl_ids].X.toarray() if hasattr(adata[selected_ctrl_ids].X, 'toarray') else adata[selected_ctrl_ids].X
        ctrl_X_tensor = torch.from_numpy(ctrl_X_data.astype(np.float32)).to(device)

        with torch.no_grad():
            pred_pert = model.sample(ctrl_X_tensor).cpu().numpy()

        true_pert = adata[selected_pert_ids].X.toarray() if hasattr(adata[selected_pert_ids].X, 'toarray') else adata[selected_pert_ids].X
        true_pert = true_pert.astype(np.float32)
        
        true_pert_pb = np.mean(true_pert, axis=0)
        pred_pert_pb = np.mean(pred_pert, axis=0)
        ctrl_pb = np.mean(ctrl_X_data, axis=0)
        
        all_true_pb.append(true_pert_pb)
        all_pred_pb.append(pred_pert_pb)
        all_ctrl_pb.append(ctrl_pb)

        metrics_results["mae"].append(compute_mae(true_pert_pb, pred_pert_pb))
        metrics_results["r2"].append(compute_r2(true_pert, pred_pert))
        metrics_results["edistance"].append(compute_edistance(true_pert, pred_pert))
        metrics_results["mmd"].append(compute_mmd(true_pert, pred_pert))
        metrics_results["pearson_all"].append(compute_pearson(true_pert_pb, pred_pert_pb))
        metrics_results["pearson_delta_all"].append(compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb))
        metrics_results["pearson_delta_de20"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=20))
        metrics_results["pearson_delta_de50"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=50))
        metrics_results["pearson_delta_de100"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=100))

        delta_true_pb = true_pert_pb - ctrl_pb
        de_genes_indices = np.argsort(np.abs(delta_true_pb))[::-1][:100]
        true_de_genes = set(adata.var_names[de_genes_indices].tolist())
        delta_pred_pb = pred_pert_pb - ctrl_pb
        pred_de_genes_indices = np.argsort(np.abs(delta_pred_pb))[::-1][:100]
        pred_de_genes = set(adata.var_names[pred_de_genes_indices].tolist())
        pred_gene_fold_changes = {gene: fc for gene, fc in zip(adata.var_names, delta_pred_pb)}
        metrics_results["des"].append(compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes))
        
        obs = pd.DataFrame({
            "perturbation_status": [f"Predicted_{pert}"] * args.n_samples,
            "origin_ctrl": selected_ctrl_ids
        }, index=[f"synthetic_{pert}_{i}" for i in range(args.n_samples)])
        var = pd.DataFrame(index=adata.var_names)
        all_synthetic_adata.append(sc.AnnData(X=pred_pert, obs=obs, var=var))


    # 5) Compute aggregate metrics across all perturbations
    print("\n" + "="*50)
    print(f"   Aggregate Evaluation Metrics (averaged over {len(perturbations)} perturbations)")
    print("="*50)

    y_true_all = np.array(all_true_pb)
    y_pred_all = np.array(all_pred_pb)

    pds_val = compute_pds(y_pred_all, y_true_all)
    print(f"Perturbation Discrimination Score (PDS): {pds_val:.4f}")

    print(f"Mean Absolute Error (MAE): {np.mean(metrics_results['mae']):.4f}")
    print(f"Differential Expression Score (DES): {np.mean(metrics_results['des']):.4f}")
    print("-" * 20)
    print(f"E-Distance: {np.mean(metrics_results['edistance']):.4f}")
    print(f"Maximum Mean Discrepancy (MMD): {np.mean(metrics_results['mmd']):.4f}")
    print(f"R-squared (R2): {np.mean(metrics_results['r2']):.4f}")
    print("-" * 20)
    print(f"Pearson (all genes): {np.mean(metrics_results['pearson_all']):.4f}")
    print(f"Pearson Delta (all genes): {np.mean(metrics_results['pearson_delta_all']):.4f}")
    print(f"Pearson Delta (top 20 DE genes): {np.mean(metrics_results['pearson_delta_de20']):.4f}")
    print(f"Pearson Delta (top 50 DE genes): {np.mean(metrics_results['pearson_delta_de50']):.4f}")
    print(f"Pearson Delta (top 100 DE genes): {np.mean(metrics_results['pearson_delta_de100']):.4f}")
    
    print("="*50 + "\nEvaluation complete!")

    # 6) UMAP Visualization with a fixed reference background
    if all_synthetic_adata:
        print("\n--- Generating UMAP Visualization ---")
        
        # Step 1: Create and process the reference AnnData for a fixed UMAP layout
        print(f"Loading training data from {args.train_data_path}...")
        adata_train = sc.read_h5ad(args.train_data_path)
        # 'adata' is the test set, already loaded
        
        adata_train.obs['data_source'] = 'train'
        adata.obs['data_source'] = 'test'
        
        adata_train.obs_names_make_unique()
        adata.obs_names_make_unique()

        # Combine train and test data to form the reference background
        adata_ref = sc.concat([adata_train, adata], join='outer', index_unique=None, fill_value=0)
        
        print("Calculating fixed UMAP embedding for reference data...")
        sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep='X', random_state=0)
        sc.tl.umap(adata_ref, random_state=0)

        # Step 2: Project the synthetic data onto the reference UMAP
        adata_synth = sc.concat(all_synthetic_adata, join='outer', index_unique=None)
        adata_synth.obs['data_source'] = 'generated'
        adata_synth.obs_names_make_unique()
        
        print("Projecting synthetic data onto the reference UMAP...")
        # sc.tl.ingest requires the .raw attribute to be set on the reference
        adata_ref.raw = adata_ref
        sc.tl.ingest(adata_synth, adata_ref, embedding_method='umap')

        # Step 3: Combine the reference and projected data for visualization
        # The 'obsm' (including X_umap) from both anndata objects are preserved during concatenation
        adata_viz = sc.concat([adata_ref, adata_synth], join='inner', index_unique=None)
        
        # --- Create a categorical column for plotting ---
        # 1. Define plot groups
        adata_viz.obs['plot_group'] = 'All Cells' # Default grey background
        
        # Identify true perturbed cells (from the test set ONLY) to color blue
        is_true_pert_mask = (adata_viz.obs['perturbation_status'] != 'Control') & (adata_viz.obs['data_source'] == 'test')
        adata_viz.obs.loc[is_true_pert_mask, 'plot_group'] = 'True Perturbed'
        
        # Identify generated cells to color orange
        is_generated_mask = adata_viz.obs['data_source'] == 'generated'
        adata_viz.obs.loc[is_generated_mask, 'plot_group'] = 'Generated Perturbed'

        # 2. Define the plotting order and colors
        category_order = [
            'All Cells', 
            'True Perturbed', 
            'Generated Perturbed'
        ]
        color_palette = {
            'All Cells': 'lightgray',
            'True Perturbed': 'blue',
            'Generated Perturbed': 'orange'
        }
        
        # 3. Make the column categorical with the specified order
        adata_viz.obs['plot_group'] = pd.Categorical(adata_viz.obs['plot_group'], categories=category_order)

        # 4. Plot using sc.pl.umap
        fig, ax = plt.subplots(figsize=(8, 6))
        point_size = 10
        
        sc.pl.umap(
            adata_viz,
            color='plot_group',
            palette=color_palette,
            ax=ax,
            size=point_size,
            title="", # Remove title
            legend_loc='best', # Place legend inside the plot
            show=False # We will handle saving and closing manually
        )
        
        os.makedirs(os.path.dirname(args.umap_plot) or ".", exist_ok=True)
        plt.savefig(args.umap_plot, dpi=300, bbox_inches='tight')
        print(f"✔ Saved UMAP plot to {args.umap_plot}")
        plt.close(fig)

    # 7) Construct and save final synthetic AnnData
    if all_synthetic_adata:
        os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
        adata_synth.write_h5ad(args.out_h5ad)
        print(f"✔ Saved combined synthetic AnnData to {args.out_h5ad}")

        # 8) Output summary
        print("\n--- Combined Synthetic AnnData Summary ---")
        print(f"Shape (cells × genes): {adata_synth.shape}")
        print("Perturbation counts:\n", adata_synth.obs['perturbation_status'].value_counts())
    else:
        print("No synthetic data was generated.")

if __name__ == "__main__":
    main()
