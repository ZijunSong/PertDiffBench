#!/usr/bin/env python3
"""
Two-stage training script:

Phase 1: Train a scVI encoder on an h5ad file and export latent representations.
Phase 2: Use the scVI latent space as input to a DDPM-MLP model
         (i.e., train DDPM in latent space) using your existing
         PairedScrnaDataset + MLPDDPMMLP + ScRNATrainer pipeline.

Example:
    python scripts/encoder_exp/scvi/train_scvi_and_latent_ddpm.py \
        --data-path data/fig1/raw_task1/task1_train_CD4T_exp.h5ad \
        --out-h5ad data/encoder_exp/task1_train_CD4T_with_scvi_latent.h5ad \
        --model-dir checkpoints/scvi_encoder/task1_CD4T \
        --n-latent 32 \
        --max-epochs 400 \
        --gpu \
        --ddpm-config configs/baselines/mlp_ddpm_mlp_latent_task1_CD4T.yaml \
        --ddpm-save-weight-dir checkpoints/mlp_ddpm_mlp_latent_task1_CD4T

Phase 2 is optional: if --ddpm-config is not provided, the script only trains scVI.
"""

import os
import argparse

import scanpy as sc
import scvi
import torch
import numpy as np

# ====== imports for Phase 2 (latent DDPM training) ======
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.scrna import PairedScrnaDataset
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP
from src.diffusion_baselines.trainers.mlp_ddpm_mlp_trainer import ScRNATrainer
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler


def summarize_adata(adata, batch_key):
    print()
    print("===== AnnData summary =====")
    print(f"Shape (cells × genes): {adata.n_obs} × {adata.n_vars}")
    print(f"type(adata.X): {type(adata.X)}")
    try:
        print(f"adata.X dtype: {adata.X.dtype}")
    except Exception:
        print("adata.X dtype: <unknown, maybe sparse matrix>")
    print(f"adata.layers keys: {list(adata.layers.keys())}")
    print(f"adata.obs keys: {list(adata.obs.columns)}")
    if batch_key is not None:
        if batch_key in adata.obs.columns:
            print(f"Batch key '{batch_key}' found in adata.obs.")
        else:
            print(f"Batch key '{batch_key}' NOT found in adata.obs.")
    print("===== End AnnData summary =====")
    print()

def train_scvi_phase(
    data_path: str,
    out_h5ad: str,
    model_dir: str,
    n_latent: int,
    max_epochs: int,
    batch_key: str = None,
    layer: str = None,
    use_gpu: bool = False,
):
    """
    Phase 1: Train (or load) scVI on input AnnData and export latent representations.
    如果 model_dir 中已有训练完毕的模型，则直接 load 而不是重新训练。

    Returns:
        adata (with obsm["X_scvi"] filled),
        latent (np.ndarray of shape [n_cells, n_latent])
    """
    print(f"Loading AnnData from: {os.path.abspath(data_path)}")
    adata = sc.read_h5ad(data_path)

    summarize_adata(adata, batch_key)

    if layer is not None and layer not in adata.layers:
        raise ValueError(
            f"Requested layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    if batch_key is not None and batch_key not in adata.obs.columns:
        print(
            f"Batch key '{batch_key}' NOT found in adata.obs. "
            f"Proceeding WITHOUT batch covariate (batch_key=None)."
        )
        batch_key = None

    scvi.settings.seed = 0
    print(f"[scVI] Seed set to {scvi.settings.seed}")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[scVI] Using device: {device}")

    print("[scVI] Setting up AnnData for scVI...")
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
    )

    model = None

    if os.path.isdir(model_dir):
        # 简单检查一下目录是否像是一个 scVI 模型目录
        has_ckpt = any(
            fname.endswith(".pt") or fname.endswith(".pth")
            for fname in os.listdir(model_dir)
        )
        if has_ckpt:
            print(
                f"[scVI] Detected existing trained model in '{model_dir}'. "
                f"Loading model instead of training from scratch."
            )
            # SCVI.load 会根据传入的 adata 重新绑定 AnnData
            model = scvi.model.SCVI.load(
                model_dir,
                adata=adata
            )

    # 如果上面没能成功找到/加载模型，则正常初始化 + 训练
    if model is None:
        print(
            f"[scVI] No existing model found in '{model_dir}'. "
            f"Initializing a new SCVI model with n_latent={n_latent}, max_epochs={max_epochs}"
        )
        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
        )

        # lightning style args
        if device == "cuda":
            accelerator = "gpu"
            devices = "auto"
        else:
            accelerator = "cpu"
            devices = "auto"

        print("[scVI] Training SCVI encoder/decoder...")
        model.train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
        )

        # 保存训练好的模型，方便下次直接 load
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir, overwrite=True)
        print(f"[scVI] ✔ Saved trained scVI model to directory: {model_dir}")
    else:
        print("[scVI] Using loaded SCVI model. Skipping training step.")

    # 无论是 load 还是新训练的，都需要算 latent
    print("[scVI] Computing latent representation with model.get_latent_representation()...")
    latent = model.get_latent_representation()
    print(f"[scVI] Latent shape: {latent.shape}")
    adata.obsm["X_scvi"] = latent

    out_dir = os.path.dirname(out_h5ad)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 保存原始表达 + latent（X 不改）
    adata.write_h5ad(out_h5ad)
    print(f"[scVI] ✔ Saved AnnData with 'X_scvi' in .obsm to: {out_h5ad}")

    print("[scVI] Phase 1 (scVI training/loading) complete.\n")
    return adata, latent

def train_ddpm_latent_phase(
    cfg_path: str,
    latent_h5ad_path: str,
    save_weight_dir_override: str = None,
    n_latent: int = None,
):
    """
    Phase 2: Train DDPM-MLP model in scVI latent space.

    We:
      1) Load a YAML config (similar to your original mlp_ddpm_mlp config)
      2) Override cfg.data.path to the latent h5ad
      3) Override cfg.model.ae.input_dim to n_latent
      4) Use PairedScrnaDataset + ScRNATrainer to train

    This is logically equivalent to:
        x -> scVI encoder -> z
        (frozen scVI)  +  DDPM(z0 -> z1_hat) + decoder
    but we precomputed z so we don't need to call scVI inside the model.
    """
    print("\n[DDPM-LATENT] Starting Phase 2 (DDPM-MLP in latent space)...")
    print(f"[DDPM-LATENT] Loading config from: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # 1) Override data.path to use the latent h5ad
    print(
        f"[DDPM-LATENT] Overriding cfg.data.path: "
        f"'{cfg.data.path}' -> '{latent_h5ad_path}'"
    )
    cfg.data.path = latent_h5ad_path

    # 2) Override latent dimensionality
    if n_latent is not None:
        print(
            f"[DDPM-LATENT] Overriding cfg.model.ae.input_dim: "
            f"'{cfg.model.ae.input_dim}' -> '{n_latent}'"
        )
        cfg.model.ae.input_dim = n_latent

    # 3) Optionally override save_weight_dir
    if save_weight_dir_override is not None:
        print(
            f"[DDPM-LATENT] Overriding cfg.train.save_weight_dir: "
            f"'{cfg.train.save_weight_dir}' -> '{save_weight_dir_override}'"
        )
        cfg.train.save_weight_dir = save_weight_dir_override

    device = torch.device(cfg.train.device)
    print(f"[DDPM-LATENT] Using device: {device}")

    # 4) Prepare dataset/dataloader
    print(f"[DDPM-LATENT] Loading latent H5AD dataset from: {os.path.abspath(latent_h5ad_path)}")
    dataset = PairedScrnaDataset(latent_h5ad_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # 5) Build model (same class as original, but now input_dim = n_latent)
    model = MLPDDPMMLP(cfg).to(device)

    # 6) Optimizer & schedulers
    optim = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched = GradualWarmupScheduler(
        optim,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=cfg.train.epoch // 10,
        after_scheduler=cosine,
    )

    trainer = ScRNATrainer(
        model,
        model.diffusion_trainer.to(device),
        optim,
        sched,
        loader,
        device,
        cfg,
    )

    # 7) Optionally skip if final checkpoint exists
    final_model_path = os.path.join(cfg.train.save_weight_dir, "model_epoch_1000.pth")
    if os.path.exists(final_model_path):
        print(
            f"[DDPM-LATENT] Found pre-trained model at '{final_model_path}'. "
            f"Skipping training."
        )
    else:
        print("[DDPM-LATENT] No pre-trained model found. Starting DDPM training...")
        trainer.train()
        print("[DDPM-LATENT] DDPM training finished.")

    print("[DDPM-LATENT] Phase 2 (latent DDPM-MLP training) complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage training: scVI encoder + DDPM-MLP in latent space."
    )
    # ===== Phase 1 (scVI) args =====
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad with raw/counts expression (or normalized, with a warning).",
    )
    parser.add_argument(
        "--out-h5ad",
        type=str,
        required=True,
        help="Output .h5ad with adata.obsm['X_scvi'] filled (original X preserved).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory to save trained scVI model.",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=32,
        help="Dimensionality of scVI latent space (n_latent).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=400,
        help="Number of training epochs for scVI.",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default=None,
        help="obs column name for batch covariate, e.g., 'batch'. "
             "If not present in adata.obs, it will be ignored.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name in adata.layers that contains count data, e.g., 'counts'. "
             "If None, uses adata.X as count matrix.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for scVI if available.",
    )

    # ===== Phase 2 (DDPM in latent) args =====
    parser.add_argument(
        "--ddpm-config",
        type=str,
        default=None,
        help="YAML config for latent DDPM-MLP. "
             "If not provided, Phase 2 will be skipped.",
    )
    parser.add_argument(
        "--ddpm-save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir for DDPM stage.",
    )
    parser.add_argument(
        "--latent-h5ad",
        type=str,
        default=None,
        help="Path to store latent-space h5ad (with X = X_scvi). "
             "If None, will use out-h5ad with suffix '_latent_X.h5ad'.",
    )

    args = parser.parse_args()

    # ---------- Phase 1: scVI ----------
    adata, latent = train_scvi_phase(
        data_path=args.data_path,
        out_h5ad=args.out_h5ad,
        model_dir=args.model_dir,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        batch_key=args.batch_key,
        layer=args.layer,
        use_gpu=args.gpu,
    )

    # # ---------- Phase 2: latent DDPM (optional) ----------
    # if args.ddpm_config is None:
    #     print("No --ddpm-config provided. Skipping Phase 2 (DDPM in latent space).")
    #     return

    # # 构造一个“latent 版本”的 h5ad：把 X 换成 X_scvi，用于 DDPM 训练
    # if args.latent_h5ad is not None:
    #     latent_h5ad_path = args.latent_h5ad
    # else:
    #     base, ext = os.path.splitext(args.out_h5ad)
    #     latent_h5ad_path = base + "_latent_X" + (ext or ".h5ad")

    # print(
    #     f"[DDPM-LATENT] Preparing latent-space h5ad for DDPM training: {latent_h5ad_path}"
    # )
    # adata_latent = adata.copy()
    # adata_latent.X = latent  # 关键一步：用 scVI latent 替换表达矩阵
    # os.makedirs(os.path.dirname(latent_h5ad_path) or ".", exist_ok=True)
    # adata_latent.write_h5ad(latent_h5ad_path)
    # print(f"[DDPM-LATENT] ✔ Saved latent-space AnnData to: {latent_h5ad_path}")

    # train_ddpm_latent_phase(
    #     cfg_path=args.ddpm_config,
    #     latent_h5ad_path=latent_h5ad_path,
    #     save_weight_dir_override=args.ddpm_save_weight_dir,
    #     n_latent=args.n_latent,
    # )


if __name__ == "__main__":
    main()
