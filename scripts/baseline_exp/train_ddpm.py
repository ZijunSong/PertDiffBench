#!/usr/bin/env python3
# scripts/train_ddpm.py

import os
import argparse
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.ddpm import DDPM
from trainers.ddpm_trainer import DDPMTrainer
from schedulers.warmup import GradualWarmupScheduler

def build_transforms(transform_cfg):
    """Build torchvision transforms pipeline from config list."""
    tfm_list = []
    for item in list(transform_cfg):
        name, params = next(iter(item.items()))
        if name == "RandomHorizontalFlip":
            tfm_list.append(transforms.RandomHorizontalFlip(**params))
        elif name == "ToTensor":
            tfm_list.append(transforms.ToTensor())
        elif name == "Normalize":
            p = OmegaConf.to_container(params, resolve=True)
            tfm_list.append(transforms.Normalize(mean=p["mean"], std=p["std"]))
        else:
            raise ValueError(f"Unknown transform: {name}")
    return transforms.Compose(tfm_list)

def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM on CIFAR-10 using a YAML config"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/ddpm_default.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # 1) Load config
    cfg = OmegaConf.load(args.config)

    # 2) Set device
    device = torch.device(cfg.train.device)

    # 3) Prepare dataset and DataLoader
    transform = build_transforms(cfg.data.transform)
    dataset = CIFAR10(
        root=cfg.data.path,
        train=True,
        download=True,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=getattr(cfg.data, "num_workers", 4),
        drop_last=True,
        pin_memory=(device.type == "cuda")
    )

    # 4) Build DDPM model
    model = DDPM(cfg).to(device)

    # 5) Set up optimizer and schedulers
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    # Use cfg.train.epoch (singular) for number of epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.train.epoch,
        eta_min=0
    )
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=max(1, cfg.train.epoch // 10),
        after_scheduler=cosine_scheduler
    )

    # 6) Instantiate trainer and launch training
    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=loader,
        cfg=cfg
    )
    trainer.train()

if __name__ == "__main__":
    main()
