#!/usr/bin/env python3
import os
import argparse

from omegaconf import OmegaConf
import torch
import numpy as np
import scanpy as sc
import pandas as pd

# eval 
from utils.metrics import (
    compute_mse,
    compute_rmse,
    compute_pearson,
    compute_kl_divergence,
)

from models.scvi_ddpm_mlp_diffusion import ScviDdpmMlp

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scVI+DDPM+MLP baseline: generate perturbed scRNA from Control and compute metrics"
    )
    parser.add_argument("-c", "--config",
                        default="configs/scvi_ddpm_mlp.yaml",
                        help="Path to the OmegaConf config")
    parser.add_argument("-k", "--ckpt",
                        default="checkpoints/scvi_ddpm_mlp/ckpt_500.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("-n", "--n_samples",
                        type=int,
                        default=100,
                        help="Number of cells to generate/evaluate")
    parser.add_argument("-o", "--out_h5ad",
                        default="samples/scvi_ddpm_mlp.h5ad",
                        help="Output path for synthetic AnnData")
    args = parser.parse_args()

    # 1) Load config and model
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)
    model = ScviDdpmMlp(cfg).to(device)
    ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state"])
    model.eval()
    print(f"✔ Loaded checkpoint from {args.ckpt}")

    # 2) Read real data and select Control / IFN cells
    adata = sc.read_h5ad(cfg.data.path)
    n = args.n_samples or cfg.sample.batch_size

    # perturbation_status, per Control / IFN 
    if "perturbation_status" in adata.obs.columns:
        ctrl_ids = adata.obs_names[adata.obs["perturbation_status"] == "Control"][:n]
        ifn_ids  = adata.obs_names[adata.obs["perturbation_status"] != "Control"][:n]
        if len(ctrl_ids) < n or len(ifn_ids) < n:
            raise ValueError(f"Not enough cells for n_samples={n}")
        # originalexpressed
        X0 = adata[ctrl_ids].X
        X_true = adata[ifn_ids].X
    else:
        # else before n as Control, after n perturbation
        X0 = adata.X[:n]
        X_true = adata.X[n:2*n]

    # 3) Tensor
    X0 = X0.toarray() if hasattr(X0, "toarray") else X0
    x0 = torch.from_numpy(X0.astype(np.float32)).to(device)

    # 4) expressed
    with torch.no_grad():
        X_pred = model.sample(x0).cpu().numpy()

    # 5) get expressedand numpy
    true_np = X_true.toarray() if hasattr(X_true, "toarray") else X_true
    true_np = true_np.astype(np.float32)

    # 6) 
    mse_val     = compute_mse(true_np, X_pred)
    rmse_val    = compute_rmse(true_np, X_pred)
    pearson_val = compute_pearson(true_np, X_pred)

    # build for KL
    p = np.clip(true_np, a_min=0, a_max=None)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-10)
    q = np.clip(X_pred, a_min=0, a_max=None)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-10)
    kl_vals = [compute_kl_divergence(p[i], q[i]) for i in range(n)]
    kl_val = float(np.mean(kl_vals))

    # evalresults
    print("\n--- Evaluation Metrics ---")
    print(f"MSE:     {mse_val:.4f}")
    print(f"RMSE:    {rmse_val:.4f}")
    print(f"Pearson: {pearson_val:.4f}")
    print(f"KL div:  {kl_val:.4f}\n")

    # 7) buildand AnnData
    gene_names = adata.var_names.tolist()
    obs = pd.DataFrame({
        "source":      ["predicted_perturb"] * n,
        "origin_ctrl": list(ctrl_ids)
    }, index=[f"synthetic_{i}" for i in range(n)])
    var = pd.DataFrame(index=gene_names)
    adata_synth = sc.AnnData(X=X_pred, obs=obs, var=var)

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    adata_synth.write_h5ad(args.out_h5ad)
    print(f"✔ Saved synthetic AnnData to {args.out_h5ad}")

    # 8) output data must
    print("\n--- Synthetic AnnData Summary ---")
    print(f"Shape (cells × genes): {adata_synth.shape}")
    print(f"Number of cells: {adata_synth.n_obs}")
    print(f"Number of genes: {adata_synth.n_vars}")
    print("obs columns:", adata_synth.obs.columns.tolist())
    print("var columns:", adata_synth.var.columns.tolist())
    print("\nFirst 5×5 expression slice:")
    print(adata_synth.X[:5, :5])
    print("\nFirst 5 obs:")
    print(adata_synth.obs.head())

if __name__ == "__main__":
    main()
