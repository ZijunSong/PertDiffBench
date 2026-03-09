#!/usr/bin/env python3
# scripts/encoder_exp/eval_geneformer_latent_ddpm_mlp.py

import os
import sys
import glob
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from omegaconf import OmegaConf

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

from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP


def resolve_ckpt(path_or_dir: str) -> str:
    """既支持直接给 .pth 文件，也支持给目录（从中选 epoch 最大的那个）."""
    if os.path.isfile(path_or_dir):
        return path_or_dir

    pattern = os.path.join(path_or_dir, "model_epoch_*.pth")
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {path_or_dir}")

    def epoch_num(p: str) -> int:
        base = os.path.basename(p)
        m = re.search(r"model_epoch_(\d+)\.pth", base)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=epoch_num)
    return ckpts[-1]


def main():
    # reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Evaluate Geneformer-latent DDPM-MLP baseline (metrics in Geneformer latent space)."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Config used to build MLPDDPMMLP.",
    )
    # 兼容不同调用习惯：-k / --ckpt / --ckpt-dir / --save-dir -> ckpt_dir
    parser.add_argument(
        "-k", "--ckpt", "--ckpt-dir", "--save-dir",
        dest="ckpt_dir",
        required=True,
        help="Path to checkpoint directory or a single .pth file.",
    )
    # 兼容 --data-path / --valid-h5ad
    parser.add_argument(
        "--data-path", "--valid-h5ad",
        dest="data_path",
        required=True,
        help="Evaluation AnnData .h5ad (must have obsm['X_geneformer'] and obs['perturbation_status']).",
    )
    parser.add_argument(
        "-n", "--n_samples",
        type=int,
        default=100,
        help="Number of control cells per perturbation to evaluate.",
    )
    parser.add_argument(
        "-o", "--out_h5ad",
        default=None,
        help="Optional: output synthetic AnnData (predicted Geneformer latent).",
    )
    parser.add_argument(
        "--latent-key",
        default="X_geneformer",
        help="obsm key for Geneformer latent.",
    )
    args = parser.parse_args()

    # 1) Load config + model
    cfg = OmegaConf.load(args.config)

    # 先直接用 config 里的 device，如果没有就自己猜
    if "train" in cfg and "device" in cfg.train:
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    print(f"[eval] Loading evaluation AnnData from: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    if "perturbation_status" not in adata.obs.columns:
        raise KeyError("adata.obs must contain 'perturbation_status' for evaluation.")
    if args.latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{args.latent_key}'] not found. Run Geneformer encoder first.")

    Z = adata.obsm[args.latent_key]
    Z = np.asarray(Z, dtype=np.float32)  # [n_cells, latent_dim]
    latent_dim = Z.shape[1]
    print(f"[eval] Geneformer latent dim = {latent_dim}")

    # 覆盖 AE input_dim，让 MLPDDPMMLP 在 Geneformer latent 空间里工作
    if "model" in cfg and "ae" in cfg.model:
        print(f"[eval] Original cfg.model.ae.input_dim = {cfg.model.ae.input_dim}")
        cfg.model.ae.input_dim = latent_dim
        print(f"[eval] Override cfg.model.ae.input_dim -> {latent_dim} for Geneformer-latent eval.")
    else:
        raise ValueError("Config must have model.ae section for MLPDDPMMLP.")

    # 构建模型 + 加载 ckpt
    model = MLPDDPMMLP(cfg).to(device)
    ckpt_path = resolve_ckpt(args.ckpt_dir)
    print(f"[eval] Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2) 基于 perturbation_status 按照 scVI 脚本的逻辑采样
    pert_status = adata.obs["perturbation_status"].astype(str)
    ctrl_mask = pert_status == "Control"
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()

    perturbations = pert_status.unique().tolist()
    perturbations = [p for p in perturbations if p != "Control"]
    print(f"[eval] Found {len(perturbations)} perturbations: {perturbations}")

    ctrl_count = len(ctrl_ids)
    pert_counts = {p: int(np.sum(pert_status == p)) for p in perturbations}
    min_pert_count = min(pert_counts.values())
    max_possible_samples = min(ctrl_count, min_pert_count)

    if args.n_samples > max_possible_samples:
        print(
            f"[eval] --n_samples ({args.n_samples}) > max_possible_samples ({max_possible_samples}). "
            f"Reduce n_samples or you will oversample."
        )
        # 这里我不直接退出，而是自动下调到 max_possible_samples，避免脚本中断
        n_samples = max_possible_samples
        print(f"[eval] Using n_samples = {n_samples}")
    else:
        n_samples = args.n_samples

    all_pred_pb, all_true_pb, all_ctrl_pb = [], [], []
    metrics_results = defaultdict(list)
    all_synthetic_adata = []

    # 为 latent 特征准备伪 var_names
    latent_var_names = np.array([f"gf_latent_{i}" for i in range(latent_dim)])

    # 3) Loop over perturbations（在 Geneformer latent 空间算指标）
    for pert in perturbations:
        print(f"\n--- Evaluating perturbation: {pert} ---")
        pert_mask = pert_status == pert
        pert_ids = adata.obs_names[pert_mask].tolist()

        selected_ctrl_ids = np.random.choice(ctrl_ids, n_samples, replace=False)
        selected_pert_ids = np.random.choice(pert_ids, n_samples, replace=False)

        ctrl_indices = adata.obs_names.get_indexer(selected_ctrl_ids)
        pert_indices = adata.obs_names.get_indexer(selected_pert_ids)

        z_ctrl = Z[ctrl_indices]   # [n_samples, latent_dim]
        z_true = Z[pert_indices]   # [n_samples, latent_dim]

        z_ctrl_tensor = torch.from_numpy(z_ctrl).to(device)

        # 模型在 Geneformer latent 空间中做 "ctrl -> pert" 映射
        with torch.no_grad():
            z_pred_tensor = model.sample(z_ctrl_tensor)  # 输出也是 [n_samples, latent_dim]
            z_pred = z_pred_tensor.cpu().numpy()

        # population means in latent space
        true_pert_pb = np.mean(z_true, axis=0)
        pred_pert_pb = np.mean(z_pred, axis=0)
        ctrl_pb = np.mean(z_ctrl, axis=0)

        all_true_pb.append(true_pert_pb)
        all_pred_pb.append(pred_pert_pb)
        all_ctrl_pb.append(ctrl_pb)

        # 各种指标都在 latent 空间算
        metrics_results["mae"].append(compute_mae(true_pert_pb, pred_pert_pb))
        metrics_results["r2"].append(compute_r2(z_true, z_pred))
        metrics_results["edistance"].append(compute_edistance(z_true, z_pred))
        metrics_results["mmd"].append(compute_mmd(z_true, z_pred))
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

        # DES: 现在的 "gene" 实际是 latent feature 名字
        delta_true_pb = true_pert_pb - ctrl_pb
        de_idx = np.argsort(np.abs(delta_true_pb))[::-1][:100]
        true_de_genes = set(latent_var_names[de_idx].tolist())

        delta_pred_pb = pred_pert_pb - ctrl_pb
        pred_de_idx = np.argsort(np.abs(delta_pred_pb))[::-1][:100]
        pred_de_genes = set(latent_var_names[pred_de_idx].tolist())
        pred_gene_fold_changes = {g: fc for g, fc in zip(latent_var_names, delta_pred_pb)}

        metrics_results["des"].append(
            compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes)
        )

        # optional synthetic AnnData（在 latent 空间）
        if args.out_h5ad:
            obs = pd.DataFrame(
                {
                    "perturbation_status": [f"Predicted_{pert}"] * n_samples,
                    "origin_ctrl": selected_ctrl_ids,
                },
                index=[f"synthetic_{pert}_{i}" for i in range(n_samples)],
            )
            var = pd.DataFrame(index=latent_var_names)
            all_synthetic_adata.append(sc.AnnData(X=z_pred, obs=obs, var=var))

    # 4) aggregate metrics & print (兼容 .sh 的 awk parser)
    print("\n" + "=" * 66)
    print(f"Aggregate Evaluation Metrics (averaged over {len(perturbations)} perturbations)")
    print("=" * 66)

    y_true_all = np.vstack(all_true_pb)
    y_pred_all = np.vstack(all_pred_pb)
    pds_val = compute_pds(y_pred_all, y_true_all)

    # 这些前缀要和 geneformer_ddpm.sh 里 awk 的正则完全一致
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
    print("=" * 66)

    # 5) 保存 synthetic AnnData（可选）
    if args.out_h5ad and len(all_synthetic_adata) > 0:
        adata_synth = sc.concat(all_synthetic_adata, join="outer", index_unique=None)
        out_path = args.out_h5ad
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        adata_synth.write_h5ad(out_path)
        print(f"[eval] Saved synthetic AnnData (latent) to: {out_path}")


if __name__ == "__main__":
    main()
