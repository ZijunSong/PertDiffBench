"""
Evaluate drug-conditioned MLP-DDPM-MLP on MOA task.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from utils.metrics import (
    compute_mae, compute_des, compute_pds,
    compute_edistance, compute_r2, compute_mmd,
    compute_pearson, compute_pearson_delta, compute_pearson_delta_de,
)
from data.scrna import get_target_drug_dose_from_test
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLPDrugCond


def load_label_encoder(path):
    data = np.load(path, allow_pickle=True)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = data["classes"]
    return le


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("-k", "--ckpt", required=True)
    parser.add_argument("--label-encoder-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("-o", "--out_h5ad", required=True)
    parser.add_argument("--gene-nums", type=int, default=None)
    parser.add_argument("--umap_plot", type=str, default="")
    parser.add_argument("--drug-key", type=str, default="perturbation")
    parser.add_argument("--dose-key", type=str, default="dose_value")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.gene_nums:
        cfg.model.ae.input_dim = args.gene_nums

    label_encoder = load_label_encoder(args.label_encoder_path)
    cfg.model.num_drug_classes = len(label_encoder.classes_)

    device = torch.device(cfg.train.device)
    model = MLPDDPMMLPDrugCond(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded", args.ckpt)

    _, drug_idx, dose_val = get_target_drug_dose_from_test(
        args.data_path, label_encoder, args.drug_key, args.dose_key
    )
    print("Target drug_idx=", drug_idx, "dose=", dose_val)

    adata = sc.read_h5ad(args.data_path)
    ctrl_mask = adata.obs["perturbation_status"] == "Control"
    pert_mask = adata.obs["perturbation_status"] == "IFN"
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()
    pert_ids = adata.obs_names[pert_mask].tolist()

    if not pert_ids:
        print("No IFN cells. Exiting.")
        sys.exit(1)

    n_samples = min(args.n_samples, len(ctrl_ids), len(pert_ids))
    selected_ctrl = np.random.choice(ctrl_ids, n_samples, replace=False)
    selected_pert = np.random.choice(pert_ids, n_samples, replace=False)

    ctrl_X = adata[selected_ctrl].X
    ctrl_X = ctrl_X.toarray() if hasattr(ctrl_X, "toarray") else ctrl_X
    ctrl_tensor = torch.from_numpy(ctrl_X.astype(np.float32)).to(device)
    drug_idx_t = torch.full((n_samples,), drug_idx, dtype=torch.long, device=device)
    dose_t = torch.full((n_samples,), dose_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_pert = model.sample(ctrl_tensor, drug_idx_t, dose_t).cpu().numpy()

    true_pert = adata[selected_pert].X
    true_pert = true_pert.toarray() if hasattr(true_pert, "toarray") else true_pert
    true_pert = true_pert.astype(np.float32)

    true_pb = np.mean(true_pert, axis=0)
    pred_pb = np.mean(pred_pert, axis=0)
    ctrl_pb = np.mean(ctrl_X, axis=0)

    mae = compute_mae(true_pb, pred_pb)
    r2 = compute_r2(true_pert, pred_pert)
    edist = compute_edistance(true_pert, pred_pert)
    mmd = compute_mmd(true_pert, pred_pert)
    p_all = compute_pearson(true_pb, pred_pb)
    pd_all = compute_pearson_delta(true_pb, pred_pb, ctrl_pb)
    pd20 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=20)
    pd50 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=50)
    pd100 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=100)

    delta_true = true_pb - ctrl_pb
    de_idx = np.argsort(np.abs(delta_true))[::-1][:100]
    true_de = set(adata.var_names[de_idx].tolist())
    delta_pred = pred_pb - ctrl_pb
    pred_de = set(adata.var_names[np.argsort(np.abs(delta_pred))[::-1][:100]].tolist())
    pred_fc = {g: fc for g, fc in zip(adata.var_names, delta_pred)}
    des = compute_des(true_de, pred_de, pred_fc)

    pds_val = compute_pds(np.array([pred_pb]), np.array([true_pb]))

    print("Perturbation Discrimination Score (PDS):", pds_val)
    print("Mean Absolute Error (MAE):", mae)
    print("Differential Expression Score (DES):", des)
    print("E-Distance:", edist)
    print("Maximum Mean Discrepancy (MMD):", mmd)
    print("R-squared (R2):", r2)
    print("Pearson (all genes):", p_all)
    print("Pearson Delta (all genes):", pd_all)
    print("Pearson Delta (top 20 DE genes):", pd20)
    print("Pearson Delta (top 50 DE genes):", pd50)
    print("Pearson Delta (top 100 DE genes):", pd100)

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    obs = pd.DataFrame({
        "perturbation_status": ["Predicted_IFN"] * n_samples,
        "origin_ctrl": selected_ctrl.tolist(),
    }, index=[f"synthetic_{i}" for i in range(n_samples)])
    var = pd.DataFrame(index=adata.var_names)
    sc.AnnData(X=pred_pert, obs=obs, var=var).write_h5ad(args.out_h5ad)
    print("Saved", args.out_h5ad)

    if args.umap_plot:
        adata_train = sc.read_h5ad(args.train_data_path)
        adata_train.obs["data_source"] = "train"
        adata.obs["data_source"] = "test"
        adata_train.obs_names_make_unique()
        adata.obs_names_make_unique()
        adata_ref = sc.concat([adata_train, adata], join="outer", index_unique=None, fill_value=0)
        sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep="X", random_state=0)
        sc.tl.umap(adata_ref, random_state=0)
        adata_synth = sc.AnnData(X=pred_pert, obs=obs, var=var)
        adata_synth.obs["data_source"] = "generated"
        adata_synth.obs_names_make_unique()
        adata_ref.raw = adata_ref
        adata_synth = adata_synth[:, adata_ref.var_names].copy()
        sc.tl.ingest(adata_synth, adata_ref, embedding_method="umap")
        adata_viz = sc.concat([adata_ref, adata_synth], join="inner", index_unique=None)
        adata_viz.obs["plot_group"] = "All Cells"
        adata_viz.obs.loc[
            (adata_viz.obs["perturbation_status"] != "Control") & (adata_viz.obs["data_source"] == "test"),
            "plot_group",
        ] = "True Perturbed"
        adata_viz.obs.loc[adata_viz.obs["data_source"] == "generated", "plot_group"] = "Generated Perturbed"
        adata_viz.obs["plot_group"] = pd.Categorical(
            adata_viz.obs["plot_group"],
            categories=["All Cells", "True Perturbed", "Generated Perturbed"],
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata_viz, color="plot_group", palette={
            "All Cells": "lightgray", "True Perturbed": "blue", "Generated Perturbed": "orange",
        }, ax=ax, size=10, show=False)
        os.makedirs(os.path.dirname(args.umap_plot) or ".", exist_ok=True)
        plt.savefig(args.umap_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved UMAP to", args.umap_plot)


if __name__ == "__main__":
    main()
