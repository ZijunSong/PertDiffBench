#!/usr/bin/env python3
import os, argparse
from omegaconf import OmegaConf

import torch
import scanpy as sc
import pandas as pd

from models.mlp_ddpm_mlp_scgpt_diff import MLPDDPMMLPscGPT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config",default="configs/mlp_ddpm_mlp_scgpt.yaml")
    p.add_argument("-k","--ckpt",required=True)
    p.add_argument("-o","--out_h5ad",default="samples/mlp_ddpm_mlp_scgpt_synthetic.h5ad")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)
    model = MLPDDPMMLPscGPT(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    # Prepare reference AnnData to get shape & var_names
    adata_ref = sc.read_h5ad(cfg.data.path)
    with torch.no_grad():
        samples = model.sample(adata_ref).cpu().numpy()

    # Save as .h5ad
    var = pd.DataFrame(index=adata_ref.var_names)
    obs = pd.DataFrame(index=[f"syn_{i}" for i in range(samples.shape[0])])
    adata_out = sc.AnnData(X=samples, obs=obs, var=var)
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    adata_out.write_h5ad(args.out_h5ad)
    print("Saved synthetic data to", args.out_h5ad)

if __name__=="__main__":
    main()
