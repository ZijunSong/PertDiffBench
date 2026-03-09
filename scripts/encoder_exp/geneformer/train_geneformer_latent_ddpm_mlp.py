#!/usr/bin/env python3
# scripts/encoder_exp/train_geneformer_latent_ddpm_mlp.py

import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

from geneformer_latent import PairedGeneformerLatentDataset
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP
from src.diffusion_baselines.trainers.mlp_ddpm_mlp_trainer import ScRNATrainer


def find_latest_ckpt(ckpt_dir: str, pattern: str = "model_epoch_*.pth"):
    ckpts = glob.glob(os.path.join(ckpt_dir, pattern))
    if not ckpts:
        return None
    ckpts.sort()
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config YAML.")
    parser.add_argument("--train-h5ad", type=str, required=True,
                        help="Encoded h5ad (with X_geneformer) for training set.")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint if available.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    # 确保 trainer 知道保存目录
    if "train" not in cfg:
        raise ValueError("Config must have a 'train' section.")
    cfg.train.save_weight_dir = args.save_dir

    # ========= 构建 dataset & dataloader =========
    train_ds = PairedGeneformerLatentDataset(args.train_h5ad, split="train")

    latent_dim = train_ds.latent.shape[1]
    print(f"[train] Geneformer latent dim = {latent_dim}")
    if "model" in cfg and "ae" in cfg.model:
        print(f"[train] Original cfg.model.ae.input_dim = {cfg.model.ae.input_dim}")
        cfg.model.ae.input_dim = latent_dim
        print(f"[train] Override cfg.model.ae.input_dim -> {latent_dim} for Geneformer-latent training.")
    else:
        raise ValueError("Config must have model.ae section for MLPDDPMMLP.")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.get("num_workers", 4),
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ========= 构建模型 =========
    # 你的 MLPDDPMMLP 是通过 cfg 初始化的 (包含 ae.input_dim / latent_dim / diffusion 等)
    model = MLPDDPMMLP(cfg).to(device)

    # ========= 优化器 & scheduler =========
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 0.0),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epoch,
    )

    # ========= Trainer =========
    trainer = ScRNATrainer(
        model=model,
        diffusion=model.diffusion_trainer,      # 你的 forward 用的是内部的 diffusion_trainer
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=train_loader,
        device=device,
        cfg=cfg,
    )

    # ========= Resume（如果需要） =========
    if args.resume:
        latest = find_latest_ckpt(args.save_dir)
        if latest is not None:
            print(f"[train] Resuming from checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            # 注意：ckpt['epoch'] 是 0-based；你的文件名是 model_epoch_{epoch+1}.pth
            trainer.current_epoch = ckpt["epoch"] + 1
            trainer.current_step = ckpt.get("step", 0)
        else:
            print("[train] --resume specified but no checkpoint found; starting from scratch.")
    else:
        print("[train] Starting training from scratch.")

    # ========= 开始训练 =========
    trainer.train()


if __name__ == "__main__":
    main()
