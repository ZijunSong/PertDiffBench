import argparse
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
import scanpy as sc
import matplotlib.pyplot as plt
import sys

# Get the project root directory to import the utils and Squidiff modules
# Note: A try-except block is safer in case __file__ is not defined (e.g., in notebooks)
try:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
except NameError:
    print("Running in an environment where __file__ is not defined. Assuming project root is accessible.")
    # You might need to adjust the path if running from a notebook in a different directory
    sys.path.insert(0, os.path.abspath('../..'))

# Ensure inner Squidiff is on path for scrna_datasets (Drug_dose_encoder)
_squidiff_inner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Squidiff')
if _squidiff_inner not in sys.path:
    sys.path.insert(1, _squidiff_inner)

from Squidiff import dist_util
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from scrna_datasets import Drug_dose_encoder
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

class sampler:
    def __init__(self, model_path, gene_size, output_dim, use_drug_structure):
        args = self.parse_args(model_path, gene_size, output_dim, use_drug_structure)
        print("Loading model and diffusion...")

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        model.load_state_dict(dist_util.load_state_dict(args['model_path']))
        model.to(dist_util.dev())
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        self.model = model
        self.diffusion = diffusion
        self.sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)

    def parse_args(self, model_path, gene_size, output_dim, use_drug_structure):
        default_args = model_and_diffusion_defaults()
        updated_args = {
            'model_path': model_path,
            'gene_size': gene_size,
            'output_dim': output_dim,
            'use_drug_structure': use_drug_structure,
            'use_ddim': True,
            'class_cond': False,
            'use_encoder': True,
            'num_layers': 3,
            'diffusion_steps': 1000,
            'batch_size': 16, # Default, can be overridden by n_samples
        }
        default_args.update(updated_args)
        return default_args

    def interp_with_direction(self, z_sem_origin, gene_size, direction, scale=1.0, add_noise_term=True):
        z_sem_origin_np = z_sem_origin.detach().cpu().numpy()
        direction_np = direction.detach().cpu().numpy()
        
        # Apply direction to the mean of the starting latent vectors
        z_sem_interp_mean = z_sem_origin_np.mean(axis=0) + direction_np * scale
        
        # Sample around the new mean point
        z_sem_interp_ = z_sem_interp_mean + np.random.randn(*z_sem_origin_np.shape) * np.std(z_sem_origin_np, axis=0)
        
        z_sem_interp_tensor = torch.tensor(z_sem_interp_, dtype=torch.float32).to(dist_util.dev())
        
        sample_interp = self.sample_fn(
            self.model,
            shape=(z_sem_origin.shape[0], gene_size),
            model_kwargs={'z_mod': z_sem_interp_tensor},
            noise=None
        )
        return sample_interp

def main():
    parser = argparse.ArgumentParser(description="Run Squidiff perturbation prediction and evaluation.")
    parser.add_argument('--model_path', type=str, default='pertbench_squidiff/model.pt')
    parser.add_argument('--data_path', type=str, default='../../data/scrna_data/scrna_positive.h5ad')
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to the training AnnData file (.h5ad) for UMAP visualization.")
    parser.add_argument('--gene_size', type=int, default=2000)
    parser.add_argument('--output_dim', type=int, default=2000)
    parser.add_argument('--use_drug_structure', action='store_true')
    parser.add_argument('--n_samples', type=int, default=0, help='Eval cells (0 = use all available paired cells)')
    parser.add_argument('--out_h5ad', type=str, default="samples/squidiff/synthetic_all_perts.h5ad")
    parser.add_argument('--umap_plot', type=str, default="samples/squidiff/umap_comparison.png")
    parser.add_argument('--control_data_path', type=str, default=None, help="Path to unified control h5ad (optional, for compatibility with fig2 scripts).")
    parser.add_argument('--seed', type=int, default=0, help="Random seed (overridden by RUN_SEED env per run)")
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    set_seed(resolve_seed(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Model Initialization ---
    device = dist_util.dev()
    print(f"Using device: {device}")
    my_sampler = sampler(args.model_path, args.gene_size, args.output_dim, args.use_drug_structure)
    my_sampler.model.to(device)

    # --- Data Loading and Preparation ---
    adata = sc.read_h5ad(args.data_path)
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()

    import sys as _sys
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from utils.max_eval_samples import resolve_eval_n_samples
    args.n_samples = resolve_eval_n_samples(args.data_path, args.n_samples, mode="multi_pert")
    print(f"Using n_samples={args.n_samples} for evaluation.")

    ctrl_mask = adata.obs['perturbation_status'] == 'Control'
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()

    # Precompute drug embeddings when use_drug_structure is True
    drug_emb_tensor = None
    if args.use_drug_structure:
        if 'smiles' not in adata.obs.columns or 'dose_value' not in adata.obs.columns:
            raise ValueError("use_drug_structure=True requires adata.obs to have 'smiles' and 'dose_value'.")
        smiles_series = adata.obs['smiles'].astype(str).replace({
            'nan': '', 'NaN': '', 'None': '', 'null': ''
        })
        dose_series = adata.obs['dose_value'].astype(str).replace({
            'nan': '0', 'NaN': '0', 'None': '0', 'null': '0'
        })
        drug_emb = Drug_dose_encoder(
            smiles_series.tolist(),
            [float(x) if x != '' else 0.0 for x in dose_series.tolist()],
            num_Bits=1024,
            comb_num=1
        )
        drug_emb_tensor = torch.tensor(drug_emb, dtype=torch.float32).to(device)

    perturbations = [p for p in adata.obs["perturbation_status"].unique() if p != "Control"]
    print(f"Found {len(perturbations)} perturbation type(s) to evaluate (non-Control conditions): {perturbations}")

    all_pred_pb, all_true_pb, all_ctrl_pb = [], [], []
    metrics_results = defaultdict(list)
    all_synthetic_adata = []

    # --- Perturbation Loop ---
    for pert in perturbations:
        print(f"\n--- Evaluating perturbation: {pert} ---")
        pert_mask = adata.obs["perturbation_status"] == pert
        pert_ids = adata.obs_names[pert_mask].tolist()

        n_use = min(len(ctrl_ids), len(pert_ids), args.n_samples)
        if n_use < 1:
            print(f"Warning: No usable cells for '{pert}'. Skipping.")
            continue
        if n_use < args.n_samples:
            print(
                f"Warning: For '{pert}', requested {args.n_samples} samples but only "
                f"{min(len(ctrl_ids), len(pert_ids))} cells available; using {n_use}."
            )

        selected_ctrl_ids = np.random.choice(ctrl_ids, n_use, replace=False)
        selected_pert_ids = np.random.choice(pert_ids, n_use, replace=False)

        # Get expression data
        ctrl_X = np.asarray(adata[selected_ctrl_ids].X)
        pert_X = np.asarray(adata[selected_pert_ids].X)

        # Encode to latent space
        if args.use_drug_structure and drug_emb_tensor is not None:
            ctrl_ix = [adata.obs_names.get_loc(cid) for cid in selected_ctrl_ids]
            pert_ix = [adata.obs_names.get_loc(pid) for pid in selected_pert_ids]
            ctrl_feat = torch.tensor(ctrl_X, dtype=torch.float32).to(device)
            pert_feat = torch.tensor(pert_X, dtype=torch.float32).to(device)
            drug_ctrl = drug_emb_tensor[ctrl_ix]
            drug_pert = drug_emb_tensor[pert_ix]
            # Control: input = [control_feature, drug_dose] per cell
            z_ctrl = my_sampler.model.encoder(
                ctrl_feat, drug_dose=drug_ctrl, control_feature=ctrl_feat
            )
            # Perturbed: use mean control expression as control_feature for direction; drug_dose from pert cells
            mean_ctrl = ctrl_feat.mean(dim=0, keepdim=True).expand(n_use, -1)
            z_pert_true = my_sampler.model.encoder(
                pert_feat, drug_dose=drug_pert, control_feature=mean_ctrl
            )
        else:
            z_ctrl = my_sampler.model.encoder(torch.tensor(ctrl_X, dtype=torch.float32).to(device))
            z_pert_true = my_sampler.model.encoder(torch.tensor(pert_X, dtype=torch.float32).to(device))
        
        # Calculate perturbation direction
        perturbation_direction = z_pert_true.mean(axis=0) - z_ctrl.mean(axis=0)

        # Predict perturbed cells
        pred_pert_tensor = my_sampler.interp_with_direction(
            z_sem_origin=z_ctrl,
            gene_size=args.gene_size,
            direction=perturbation_direction,
            scale=1.0
        )
        pred_pert = pred_pert_tensor.detach().cpu().numpy()

        # Create pseudobulk profiles
        true_pert_pb = np.mean(pert_X, axis=0)
        pred_pert_pb = np.mean(pred_pert, axis=0)
        ctrl_pb = np.mean(ctrl_X, axis=0)
        
        all_true_pb.append(true_pert_pb)
        all_pred_pb.append(pred_pert_pb)
        all_ctrl_pb.append(ctrl_pb)

        # Calculate and store metrics
        metrics_results["mae"].append(compute_mae(true_pert_pb, pred_pert_pb))
        metrics_results["r2"].append(compute_r2(pert_X, pred_pert))
        metrics_results["edistance"].append(compute_edistance(pert_X, pred_pert))
        metrics_results["mmd"].append(compute_mmd(pert_X, pred_pert))
        metrics_results["pearson_all"].append(compute_pearson(true_pert_pb, pred_pert_pb))
        metrics_results["pearson_delta_all"].append(compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb))
        metrics_results["pearson_delta_de20"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=20))
        metrics_results["pearson_delta_de50"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=50))
        metrics_results["pearson_delta_de100"].append(compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=100))
        
        # DES calculation
        delta_true_pb = true_pert_pb - ctrl_pb
        de_genes_indices = np.argsort(np.abs(delta_true_pb))[::-1][:100]
        true_de_genes = set(adata.var_names[de_genes_indices].tolist())
        delta_pred_pb = pred_pert_pb - ctrl_pb
        pred_de_genes_indices = np.argsort(np.abs(delta_pred_pb))[::-1][:100]
        pred_de_genes = set(adata.var_names[pred_de_genes_indices].tolist())
        pred_gene_fold_changes = {gene: fc for gene, fc in zip(adata.var_names, delta_pred_pb)}
        metrics_results["des"].append(compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes))

        # Store synthetic data
        obs = pd.DataFrame({
            "perturbation_status": [f"Predicted_{pert}"] * n_use
        }, index=[f"synthetic_{pert}_{i}" for i in range(n_use)])
        all_synthetic_adata.append(sc.AnnData(X=pred_pert, obs=obs, var=adata.var.copy()))

    # --- Aggregate Metrics ---
    print("\n" + "="*50)
    print(f"   Aggregate Evaluation Metrics (averaged over {len(perturbations)} perturbations)")
    print("="*50)

    y_true_all = np.array(all_true_pb)
    y_pred_all = np.array(all_pred_pb)

    print(f"Perturbation Discrimination Score (PDS): {compute_pds(y_pred_all, y_true_all):.4f}")
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

    # # --- UMAP Visualization and Saving ---
    # if all_synthetic_adata:
    #     print("\n--- Generating UMAP Visualization ---")
        
    #     # Step 1: Create and process the reference AnnData for a fixed UMAP layout
    #     print(f"Loading training data from {args.train_data_path}...")
    #     adata_train = sc.read_h5ad(args.train_data_path)
    #     # 'adata' is the test set, already loaded
        
    #     adata_train.obs['data_source'] = 'train'
    #     adata.obs['data_source'] = 'test'
        
    #     adata_train.obs_names_make_unique()
    #     adata.obs_names_make_unique()

    #     # Combine train and test data to form the reference background
    #     adata_ref = sc.concat([adata_train, adata], join='outer', index_unique=None, fill_value=0)
        
    #     print("Calculating fixed UMAP embedding for reference data...")
    #     sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep='X', random_state=0)
    #     sc.tl.umap(adata_ref, random_state=0)

    #     # Step 2: Project the synthetic data onto the reference UMAP
    #     adata_synth = sc.concat(all_synthetic_adata, join='outer', index_unique=None)
    #     adata_synth.obs['data_source'] = 'generated'
    #     adata_synth.obs_names_make_unique()
        
    #     print("Projecting synthetic data onto the reference UMAP...")
    #     # sc.tl.ingest requires the .raw attribute to be set on the reference
    #     adata_ref.raw = adata_ref
    #     adata_synth = adata_synth[:, adata_ref.var_names].copy()
    #     sc.tl.ingest(adata_synth, adata_ref, embedding_method='umap')

    #     # Step 3: Combine the reference and projected data for visualization
    #     adata_viz = sc.concat([adata_ref, adata_synth], join='inner', index_unique=None)
        
    #     # --- Create a categorical column for plotting ---
    #     # 1. Define plot groups
    #     adata_viz.obs['plot_group'] = 'All Cells' # Default grey background
        
    #     # Identify true perturbed cells (from the test set ONLY) to color blue
    #     is_true_pert_mask = (adata_viz.obs['perturbation_status'] != 'Control') & (adata_viz.obs['data_source'] == 'test')
    #     adata_viz.obs.loc[is_true_pert_mask, 'plot_group'] = 'True Perturbed'
        
    #     # Identify generated cells to color orange
    #     is_generated_mask = adata_viz.obs['data_source'] == 'generated'
    #     adata_viz.obs.loc[is_generated_mask, 'plot_group'] = 'Generated Perturbed'

    #     # 2. Define the plotting order and colors
    #     category_order = [
    #         'All Cells', 
    #         'True Perturbed', 
    #         'Generated Perturbed'
    #     ]
    #     color_palette = {
    #         'All Cells': 'lightgray',
    #         'True Perturbed': 'blue',
    #         'Generated Perturbed': 'orange'
    #     }
        
    #     # 3. Make the column categorical with the specified order
    #     adata_viz.obs['plot_group'] = pd.Categorical(adata_viz.obs['plot_group'], categories=category_order)

    #     # 4. Plot using sc.pl.umap
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     sc.pl.umap(
    #         adata_viz,
    #         color='plot_group',
    #         palette=color_palette,
    #         ax=ax,
    #         size=10,
    #         title="",
    #         legend_loc='best',
    #         show=False
    #     )
        
    #     # Save plot and data
    #     os.makedirs(os.path.dirname(args.umap_plot), exist_ok=True, mode=0o755)
    #     plt.savefig(args.umap_plot, dpi=300, bbox_inches='tight')
    #     print(f"✔ Saved UMAP plot to {args.umap_plot}")
    #     plt.close(fig)

    #     os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True, mode=0o755)
    #     adata_synth.write_h5ad(args.out_h5ad)
    #     print(f"✔ Saved combined synthetic AnnData to {args.out_h5ad}")
    # else:
    #     print("No synthetic data was generated.")

if __name__ == "__main__":
    main()
