#!/usr/bin/env python3
# scripts/encoder_exp/scimilarity/eval_scimilarity_latent_ddpm_mlp.py

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import scanpy as sc
import pandas as pd

from omegaconf import OmegaConf
import matplotlib.pyplot as plt  # 只为了保持接口一致，如果你不用画图可以删掉

# Metrics (和你 eval_mlp_ddpm_mlp.py 一致)
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

# 模型：latent DDPM + decoder
from src.diffusion_baselines.models.scimilarity_latent_ddpm_mlp import (
    ScimilarityLatentDDPMMLP,
)


def infer_latent_and_hidden_dim_from_ckpt(ckpt_state_dict):
    """
    从 checkpoint 的 state_dict 自动推断 latent_dim / hidden_dim。

    利用 decoder 第一层:
        decoder.net.0: Linear(latent_dim -> hidden_dim)
        weight 形状为 [hidden_dim, latent_dim]
    """
    for name, param in ckpt_state_dict.items():
        if name.endswith("decoder.net.0.weight"):
            w = param
            hidden_dim, latent_dim = w.shape
            print(
                f"[eval] Inferred hidden_dim={hidden_dim}, latent_dim={latent_dim} "
                f"from parameter '{name}'"
            )
            return int(latent_dim), int(hidden_dim)

    raise RuntimeError(
        "Could not find 'decoder.net.0.weight' in checkpoint to infer latent_dim. "
        "Please check the model definition or adjust this helper."
    )


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Evaluate SCimilarity-latent DDPM+MLP decoder on scRNA-seq."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/baselines/scimilarity_ddpm_mlp.yaml",
        help="Path to the OmegaConf config YAML.",
    )
    parser.add_argument(
        "-k",
        "--ckpt",
        required=True,
        help="Path to trained model checkpoint (model_final.pth).",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=100,
        help="Number of cells per perturbation for evaluation.",
    )
    parser.add_argument(
        "-o",
        "--out_h5ad",
        default="samples/encoder_exp/scimilarity_ddpm/scim_latent_ddpm_mlp_eval.h5ad",
        help="Output path for synthetic AnnData.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the VALID h5ad WITH SCimilarity latent (apply_scimilarity_encoder 输出).",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="X_scim",
        help="Key in adata.obsm containing SCimilarity latent embeddings.",
    )
    args = parser.parse_args()

    # 1) Load config (只用里面的 model / train，不再假设有 cfg.data)
    cfg = OmegaConf.load(args.config)

    # 2) Set device
    if "train" in cfg and "device" in cfg.train:
        device_str = cfg.train.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[eval] Using device: {device}")

    # 3) Load checkpoint first, infer latent_dim / hidden_dim
    print(f"[eval] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    latent_dim, hidden_dim = infer_latent_and_hidden_dim_from_ckpt(state_dict)

    # 覆盖 config 里的 latent_dim / hidden_dim，使之与 checkpoint 完全一致
    old_latent = cfg.model.ae.latent_dim
    old_hidden = cfg.model.ae.hidden_dim
    cfg.model.ae.latent_dim = latent_dim
    cfg.model.ae.hidden_dim = hidden_dim

    print(
        f"[eval] Overriding cfg.model.ae.latent_dim: {old_latent} -> {latent_dim}"
    )
    print(
        f"[eval] Overriding cfg.model.ae.hidden_dim: {old_hidden} -> {hidden_dim}"
    )

    # 4) Build model with updated cfg, then load state_dict
    model = ScimilarityLatentDDPMMLP(cfg).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[eval][Warn] Missing keys when loading: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[eval][Warn] Unexpected keys when loading: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    model.eval()
    print("[eval] ✔ model_state_dict loaded (with inferred dims).")

    # 5) Load VALID data (with latent) and prepare evaluation
    print(f"[eval] Reading VALID AnnData with latent from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    if args.latent_key not in adata.obsm:
        raise KeyError(
            f"Latent key '{args.latent_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    X_latent = adata.obsm[args.latent_key]  # shape: [N_cells, latent_dim]
    if X_latent.shape[1] != latent_dim:
        raise RuntimeError(
            f"Latent dimension mismatch: adata.obsm['{args.latent_key}'] has dim {X_latent.shape[1]}, "
            f"but checkpoint/model expects {latent_dim}."
        )

    # 可选：如果你担心 gene_dim 也不一致，可以在这里 assert 一下：
    gene_dim_cfg = cfg.model.ae.input_dim
    if gene_dim_cfg != adata.n_vars:
        print(
            f"[eval][Warn] cfg.model.ae.input_dim={gene_dim_cfg} "
            f"!= adata.n_vars={adata.n_vars}. "
            f"Decoder最后一层可能和数据不匹配（除非训练时就是这样）。"
        )

    ctrl_mask = adata.obs["perturbation_status"] == "Control"
    ctrl_ids = adata.obs_names[ctrl_mask].tolist()

    perturbations = adata.obs["perturbation_status"].unique().tolist()
    perturbations = [p for p in perturbations if p != "Control"]
    print(f"[eval] Found {len(perturbations)} perturbations in VALID set: {perturbations}")

    # 预检查样本数
    print("\n--- Checking sample counts in VALID set ---")
    ctrl_count = len(ctrl_ids)
    pert_counts = {p: int(np.sum(adata.obs["perturbation_status"] == p)) for p in perturbations}

    if not pert_counts:
        print("Warning: No perturbation groups found in the VALID data. Exiting.")
        return

    min_pert_count = min(pert_counts.values())
    max_possible_samples = min(ctrl_count, min_pert_count)

    print(f"Control cells available: {ctrl_count}")
    print(f"Minimum cells in a perturbation group: {min_pert_count}")
    print(f"Maximum possible --n_samples: {max_possible_samples}")

    if args.n_samples > max_possible_samples:
        print(
            f"\nError: --n_samples ({args.n_samples}) exceeds the maximum possible value ({max_possible_samples})."
        )
        sys.exit(1)

    all_pred_pb = []
    all_true_pb = []
    all_ctrl_pb = []
    metrics_results = defaultdict(list)
    all_synthetic_adata = []

    # 6) 对每个 perturbation 评估
    for pert in perturbations:
        print(f"\n--- Evaluating perturbation: {pert} ---")
        pert_mask = adata.obs["perturbation_status"] == pert
        pert_ids = adata.obs_names[pert_mask].tolist()

        selected_ctrl_ids = np.random.choice(ctrl_ids, args.n_samples, replace=False)
        selected_pert_ids = np.random.choice(pert_ids, args.n_samples, replace=False)

        # 控制组 latent 作为模型输入
        ctrl_latent = X_latent[adata.obs_names.get_indexer(selected_ctrl_ids)]
        ctrl_latent_tensor = torch.from_numpy(
            ctrl_latent.astype(np.float32)
        ).to(device)

        # 真实扰动表达，用于 metric（表达空间）
        true_pert = adata[selected_pert_ids].X
        if hasattr(true_pert, "toarray"):
            true_pert = true_pert.toarray()
        true_pert = true_pert.astype(np.float32)

        # 真实 control 表达（计算 delta / ctrl_pb）
        ctrl_expr = adata[selected_ctrl_ids].X
        if hasattr(ctrl_expr, "toarray"):
            ctrl_expr = ctrl_expr.toarray()
        ctrl_expr = ctrl_expr.astype(np.float32)

        # 通过 latent DDPM+decoder 生成预测扰动表达
        with torch.no_grad():
            pred_pert = model.sample_from_latent(ctrl_latent_tensor).cpu().numpy().astype(np.float32)

        # 7) 计算各种 metrics（和 MLP baseline 一致）
        true_pert_pb = np.mean(true_pert, axis=0)
        pred_pert_pb = np.mean(pred_pert, axis=0)
        ctrl_pb = np.mean(ctrl_expr, axis=0)

        all_true_pb.append(true_pert_pb)
        all_pred_pb.append(pred_pert_pb)
        all_ctrl_pb.append(ctrl_pb)

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

        # DES
        delta_true_pb = true_pert_pb - ctrl_pb
        de_genes_indices = np.argsort(np.abs(delta_true_pb))[::-1][:100]
        true_de_genes = set(adata.var_names[de_genes_indices].tolist())

        delta_pred_pb = pred_pert_pb - ctrl_pb
        pred_de_genes_indices = np.argsort(np.abs(delta_pred_pb))[::-1][:100]
        pred_de_genes = set(adata.var_names[pred_de_genes_indices].tolist())
        pred_gene_fold_changes = {
            gene: fc for gene, fc in zip(adata.var_names, delta_pred_pb)
        }
        metrics_results["des"].append(
            compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes)
        )

        # 保存 synthetic AnnData 方便后续 concat
        obs = pd.DataFrame(
            {
                "perturbation_status": [f"Predicted_{pert}"] * args.n_samples,
                "origin_ctrl": selected_ctrl_ids,
            },
            index=[f"synthetic_{pert}_{i}" for i in range(args.n_samples)],
        )
        var = pd.DataFrame(index=adata.var_names)
        all_synthetic_adata.append(sc.AnnData(X=pred_pert, obs=obs, var=var))

    # 8) 聚合指标并打印
    print("\n" + "=" * 50)
    print(
        f"   Aggregate Evaluation Metrics (averaged over {len(perturbations)} perturbations)"
    )
    print("=" * 50)

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
    print(
        f"Pearson Delta (all genes): {np.mean(metrics_results['pearson_delta_all']):.4f}"
    )
    print(
        f"Pearson Delta (top 20 DE genes): {np.mean(metrics_results['pearson_delta_de20']):.4f}"
    )
    print(
        f"Pearson Delta (top 50 DE genes): {np.mean(metrics_results['pearson_delta_de50']):.4f}"
    )
    print(
        f"Pearson Delta (top 100 DE genes): {np.mean(metrics_results['pearson_delta_de100']):.4f}"
    )
    print("=" * 50 + "\nEvaluation complete!")

    # 9) 合并并保存 synthetic AnnData（可选）
    if all_synthetic_adata:
        os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
        adata_synth = sc.concat(all_synthetic_adata, join="outer", index_unique=None)
        adata_synth.write_h5ad(args.out_h5ad)
        print(f"✔ Saved combined synthetic AnnData to {args.out_h5ad}")
        print("\n--- Combined Synthetic AnnData Summary ---")
        print(f"Shape (cells × genes): {adata_synth.shape}")
        print(
            "Perturbation counts:\n",
            adata_synth.obs["perturbation_status"].value_counts(),
        )
    else:
        print("No synthetic data was generated.")


if __name__ == "__main__":
    main()
