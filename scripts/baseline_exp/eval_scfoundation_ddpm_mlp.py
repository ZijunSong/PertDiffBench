#!/usr/bin/env python3
import os, argparse
from omegaconf import OmegaConf

import torch
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse

from src.diffusion_baselines.models.scfoundation_ddpm_mlp_diffusion import LatentDiffusionModel
from utils.metrics import compute_mse, compute_rmse, compute_pearson, compute_kl_divergence

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config",default="configs/baselines/scfoundation_ddpm_mlp.yaml", help="Path to the config file.")
    p.add_argument("-k","--ckpt",default="checkpoints/scfoundation/ckpt_100.pt", help="Path to the model checkpoint file.")
    p.add_argument("-o","--out_h5ad",default="results/scfoundation_ddpm_mlp.h5ad", help="Path to save the output .h5ad file.")
    # 1. *** 主要修改：将 default 修改为 None ***
    p.add_argument("-n", "--num_samples", type=int, default=None, help="Number of samples to generate. If not provided, defaults to the number of cells in the reference data.")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    model = LatentDiffusionModel(cfg).to(device)
    
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    adata_ref = sc.read_h5ad(cfg.data.path)

    if args.num_samples is None:
        num_to_generate = adata_ref.shape[0]
        print(f"Number of samples not set. Defaulting to reference data size: {num_to_generate}")
    else:
        num_to_generate = args.num_samples

    print(f"Generating {num_to_generate} synthetic samples...")
    with torch.no_grad():
        y_pred = model.sample(
            num_samples=num_to_generate,
            device=device
        ).cpu().numpy()

    y_true = adata_ref.X.toarray() if issparse(adata_ref.X) else adata_ref.X

    if y_pred.shape == y_true.shape:
        print("\n" + "="*50)
        print("Shapes match. Running evaluations against reference data...")
        
        y_true_mean = y_true.mean(axis=0)
        y_pred_mean = y_pred.mean(axis=0)
        
        mse_val = compute_mse(y_true_mean, y_pred_mean)
        rmse_val = compute_rmse(y_true_mean, y_pred_mean)
        pearson_val = compute_pearson(y_true, y_pred)
        kl_div_val = compute_kl_divergence(y_true_mean, y_pred_mean)
        
        print("\n--- Evaluation Results ---")
        print(f"  MSE (on mean expression): {mse_val:.6f}")
        print(f"  RMSE (on mean expression): {rmse_val:.6f}")
        print(f"  Pearson Correlation (all data): {pearson_val:.6f}")
        print(f"  KL Divergence (on mean expression): {kl_div_val:.6f}")
        print("--------------------------\n")
    else:
        print("\n" + "="*50)
        print(f"[WARN] Shape of generated data {y_pred.shape} does not match reference data {y_true.shape}.")
        print("Skipping evaluation metrics.")
        print("="*50 + "\n")

    print(f"Saving synthetic data with shape {y_pred.shape}...")
    var = pd.DataFrame(index=adata_ref.var_names)
    obs = pd.DataFrame(index=[f"syn_{i}" for i in range(y_pred.shape[0])])
    adata_out = sc.AnnData(X=y_pred, obs=obs, var=var)
    
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    adata_out.write_h5ad(args.out_h5ad)
    print(f"✅ Saved synthetic data to {args.out_h5ad}")

if __name__=="__main__":
    main()