#!/usr/bin/env python3

import os
import sys
import argparse
from collections import defaultdict

from omegaconf import OmegaConf
import torch
import numpy as np
import pandas as pd
import scanpy as sc

from utils.metrics import (
    compute_mae,
    compute_des,
    compute_pds,
    compute_edistance,
    compute_r2,
    compute_mmd,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_de,
    compute_pearson_delta_de,
)

from src.diffusion_baselines.models.scvi_latent_ddpm_mlp import ScviLatentDDPMMLP


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Evaluate scVI-latent DDPM-MLP baseline on scRNA-seq (metrics in gene space)."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Config used to build ScviLatentDDPMMLP.",
    )
    parser.add_argument(
        "-k", "--ckpt",
        required=True,
        help="Path to trained ScviLatentDDPMMLP checkpoint (.pth).",
    )
    parser.add_argument(
        "-n", "--n_samples",
        type=int,
        default=100,
        help="Number of control cells to generate and evaluate per perturbation.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Test AnnData .h5ad (must have latent embeddings in adata.obsm[obsm_key]).",
    )
    parser.add_argument(
        "--obsm-key",
        default="X_tx1",
        help="Latent key in adata.obsm (e.g., X_tx1 or X_scvi).",
    )

    parser.add_argument(
        "-o", "--out_h5ad",
        default=None,
        help="Optional: output synthetic AnnData path.",
    )
    args = parser.parse_args()

    # 1) Load config + model
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    model = ScviLatentDDPMMLP(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    # 2) Load test AnnData
    print(f"Loading evaluation AnnData from: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    if "perturbation_status" not in adata.obs.columns:
        raise KeyError("adata.obs must contain 'perturbation_status'.")
    if args.obsm_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.obsm_key}'] not found. Run encoder on test set first.")

    X = adata.X
    Z = adata.obsm[args.obsm_key]

    ctrl_mask = adata.obs["perturbation_status"] == "Control"
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()

    perturbations = adata.obs["perturbation_status"].unique().tolist()
    perturbations = [p for p in perturbations if p != "Control"]
    print(f"Found {len(perturbations)} perturbations in test set: {perturbations}")

    # sample count check
    ctrl_count = len(ctrl_ids)
    pert_counts = {p: int(np.sum(adata.obs["perturbation_status"] == p)) for p in perturbations}
    min_pert_count = min(pert_counts.values())
    max_possible_samples = min(ctrl_count, min_pert_count)

    if args.n_samples > max_possible_samples:
        print(
            f"--n_samples ({args.n_samples}) > max_possible_samples ({max_possible_samples}). "
            f"Please reduce n_samples."
        )
        sys.exit(1)

    all_pred_pb, all_true_pb, all_ctrl_pb = [], [], []
    metrics_results = defaultdict(list)
    all_synthetic_adata = []

    # 3) Loop over perturbations
    for pert in perturbations:
        print(f"\n--- Evaluating perturbation: {pert} ---")
        pert_mask = adata.obs["perturbation_status"] == pert
        pert_ids = adata.obs_names[pert_mask].tolist()

        selected_ctrl_ids = np.random.choice(ctrl_ids, args.n_samples, replace=False)
        selected_pert_ids = np.random.choice(pert_ids, args.n_samples, replace=False)

        # control latent / gene
        ctrl_indices = adata.obs_names.get_indexer(selected_ctrl_ids)
        z0 = Z[ctrl_indices]                      # [n_samples, latent_dim]
        ctrl_X = X[ctrl_indices]

        z0_tensor = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)

        # predicted perturbed (gene space)
        with torch.no_grad():
            pred_pert_tensor = model.sample_from_latent(z0_tensor)  # [n_samples, G]
            pred_pert = pred_pert_tensor.cpu().numpy()

        # true perturbed (gene space)
        pert_indices = adata.obs_names.get_indexer(selected_pert_ids)
        true_pert_X = X[pert_indices]
        true_pert = true_pert_X.toarray() if hasattr(true_pert_X, "toarray") else true_pert_X
        true_pert = np.asarray(true_pert, dtype=np.float32)

        ctrl_X_data = ctrl_X.toarray() if hasattr(ctrl_X, "toarray") else ctrl_X
        ctrl_X_data = np.asarray(ctrl_X_data, dtype=np.float32)

        # population means in gene space
        true_pert_pb = np.mean(true_pert, axis=0)
        pred_pert_pb = np.mean(pred_pert, axis=0)
        ctrl_pb = np.mean(ctrl_X_data, axis=0)

        all_true_pb.append(true_pert_pb)
        all_pred_pb.append(pred_pert_pb)
        all_ctrl_pb.append(ctrl_pb)

        # metrics (全部在 gene 空间)
        metrics_results["mae"].append(compute_mae(true_pert_pb, pred_pert_pb))
        metrics_results["r2"].append(compute_r2(true_pert, pred_pert))
        metrics_results["edistance"].append(compute_edistance(true_pert, pred_pert))
        metrics_results["mmd"].append(compute_mmd(true_pert, pred_pert))
        metrics_results["pearson_all"].append(compute_pearson(true_pert_pb, pred_pert_pb))
        metrics_results["pearson_delta_all"].append(
            compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb)
        )
        metrics_results["pearson_delta_de20"].append(
            compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=20)
        )
        metrics_results["pearson_delta_de50"].append(
            compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=50)
        )
        metrics_results["pearson_delta_de100"].append(
            compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=100)
        )

        # DES (gene 空间)
        delta_true_pb = true_pert_pb - ctrl_pb
        de_genes_indices = np.argsort(np.abs(delta_true_pb))[::-1][:100]
        true_de_genes = set(adata.var_names[de_genes_indices].tolist())

        delta_pred_pb = pred_pert_pb - ctrl_pb
        pred_de_genes_indices = np.argsort(np.abs(delta_pred_pb))[::-1][:100]
        pred_de_genes = set(adata.var_names[pred_de_genes_indices].tolist())
        pred_gene_fold_changes = {gene: fc for gene, fc in zip(adata.var_names, delta_pred_pb)}

        metrics_results["des"].append(
            compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes)
        )

        # optional: 保存 synthetic AnnData
        if args.out_h5ad:
            obs = pd.DataFrame(
                {
                    "perturbation_status": [f"Predicted_{pert}"] * args.n_samples,
                    "origin_ctrl": selected_ctrl_ids,
                },
                index=[f"synthetic_{pert}_{i}" for i in range(args.n_samples)],
            )
            var = pd.DataFrame(index=adata.var_names)
            all_synthetic_adata.append(sc.AnnData(X=pred_pert, obs=obs, var=var))

    # 4) aggregate metrics
    print("\n" + "=" * 50)
    print(f"Aggregate metrics over {len(perturbations)} perturbations")
    print("=" * 50)

    y_true_all = np.vstack(all_true_pb)
    y_pred_all = np.vstack(all_pred_pb)
    pds_val = compute_pds(y_pred_all, y_true_all)
    print(f"PDS: {pds_val:.4f}")
    print(f"MAE: {np.mean(metrics_results['mae']):.4f}")
    print(f"DES: {np.mean(metrics_results['des']):.4f}")
    print("-" * 20)
    print(f"E-distance: {np.mean(metrics_results['edistance']):.4f}")
    print(f"MMD:        {np.mean(metrics_results['mmd']):.4f}")
    print(f"R2:         {np.mean(metrics_results['r2']):.4f}")
    print("-" * 20)
    print(f"Pearson(all genes):              {np.mean(metrics_results['pearson_all']):.4f}")
    print(f"Pearson Delta(all genes):        {np.mean(metrics_results['pearson_delta_all']):.4f}")
    print(f"Pearson Delta(top 20 DE genes):  {np.mean(metrics_results['pearson_delta_de20']):.4f}")
    print(f"Pearson Delta(top 50 DE genes):  {np.mean(metrics_results['pearson_delta_de50']):.4f}")
    print(f"Pearson Delta(top 100 DE genes): {np.mean(metrics_results['pearson_delta_de100']):.4f}")
    print("=" * 50)

    # 保存 synthetic AnnData（可选）
    if args.out_h5ad and len(all_synthetic_adata) > 0:
        adata_synth = sc.concat(all_synthetic_adata, join="outer", index_unique=None)
        out_path = args.out_h5ad
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        adata_synth.write_h5ad(out_path)
        print(f"Saved synthetic AnnData to: {out_path}")


if __name__ == "__main__":
    main()