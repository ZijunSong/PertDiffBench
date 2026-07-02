#!/usr/bin/env python3
"""
Evaluate a trained ScviLatentDDPMMLP checkpoint when latents come from Tahoe-x1 (Tx1) embeddings.

This mirrors eval_scvi_latent_ddpm_mlp.py, but:
- lets you choose which latent key to read from adata.obsm (default: X_tx1)
- keeps everything else the same: metrics computed in gene space

Example:
  python scripts/encoder_exp/eval_tx1_latent_ddpm_mlp.py \
      -c configs/baselines/tx1_ddpm_mlp.yaml \
      -k checkpoints/tx1_ddpm/latent_ddpm/model_final.pth \
      --data-path data/fig1/raw_task1/task1_test_CD4T_with_tx1_latent.h5ad \
      --obsm-key X_tx1 \
      -n 200 \
      -o outputs/synth_tx1.h5ad
"""

from __future__ import annotations

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
    compute_r2,
    compute_pearson_delta,
)

from src.diffusion_baselines.models.scvi_latent_ddpm_mlp import ScviLatentDDPMMLP


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM+decoder in latent space (Tx1/scVI/etc).")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config for ScviLatentDDPMMLP (only used to build the model).",
    )
    parser.add_argument(
        "-k", "--ckpt",
        required=True,
        help="Path to trained checkpoint (.pth).",
    )
    parser.add_argument(
        "-n", "--n_samples",
        type=int,
        default=0,
        help="Number of control cells to generate and evaluate per perturbation.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Test AnnData (.h5ad). Must contain adata.obsm[obsm_key] and obs['perturbation_status'].",
    )
    parser.add_argument(
        "--obsm-key",
        type=str,
        default="X_tx1",
        help="Which adata.obsm key stores the latent embeddings (e.g., X_tx1 or X_scvi).",
    )
    parser.add_argument(
        "-o", "--out_h5ad",
        default=None,
        help="Optional: output synthetic AnnData path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overridden by RUN_SEED env per run)")
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    set_seed(resolve_seed(getattr(args, "seed", 0)))

    # 1) Load model
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

    # Sanity check on sample counts
    ctrl_count = len(ctrl_ids)
    pert_counts = {p: int(np.sum(adata.obs["perturbation_status"] == p)) for p in perturbations}
    min_pert_count = min(pert_counts.values()) if len(pert_counts) else 0
    max_possible_samples = min(ctrl_count, min_pert_count) if len(pert_counts) else 0

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

        # sample matching number of control and perturbed cells
        selected_ctrl_ids = np.random.choice(ctrl_ids, size=args.n_samples, replace=False).tolist()
        selected_pert_ids = np.random.choice(pert_ids, size=args.n_samples, replace=False).tolist()

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

        # metrics
        metrics_results["mae"].append(compute_mae(true_pert_pb, pred_pert_pb))
        metrics_results["r2"].append(compute_r2(true_pert, pred_pert))
        metrics_results["des"].append(compute_des(true_pert_pb, pred_pert_pb, ctrl_pb))
        metrics_results["pearson_delta_de50"].append(compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb, top_k=50))
        metrics_results["pearson_delta_de100"].append(compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb, top_k=100))

        print(
            f"mae={metrics_results['mae'][-1]:.4f}  "
            f"r2={metrics_results['r2'][-1]:.4f}  "
            f"des={metrics_results['des'][-1]:.4f}  "
            f"pearsonΔ(de50)={metrics_results['pearson_delta_de50'][-1]:.4f}"
        )

        # optional synthetic AnnData
        if args.out_h5ad:
            adata_s = sc.AnnData(
                X=pred_pert,
                obs=pd.DataFrame(
                    {
                        "perturbation_status": [pert] * args.n_samples,
                        "source": ["pred"] * args.n_samples,
                    },
                    index=selected_ctrl_ids,
                ),
                var=adata.var.copy(),
            )
            all_synthetic_adata.append(adata_s)

    # 4) Summary
    print("\n" + "=" * 50)
    print(f"MAE:                      {np.mean(metrics_results['mae']):.4f}")
    print(f"R2 (cell-wise):           {np.mean(metrics_results['r2']):.4f}")
    print(f"DES:                      {np.mean(metrics_results['des']):.4f}")
    print(f"Pearson Delta(top 50 DE): {np.mean(metrics_results['pearson_delta_de50']):.4f}")
    print(f"Pearson Delta(top 100 DE):{np.mean(metrics_results['pearson_delta_de100']):.4f}")
    print("=" * 50)

    if args.out_h5ad and len(all_synthetic_adata) > 0:
        adata_synth = sc.concat(all_synthetic_adata, join="outer", index_unique=None)
        out_path = args.out_h5ad
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        adata_synth.write_h5ad(out_path)
        print(f"Saved synthetic AnnData to: {out_path}")


if __name__ == "__main__":
    main()
