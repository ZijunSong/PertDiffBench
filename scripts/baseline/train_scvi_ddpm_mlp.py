#!/usr/bin/env python3
import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.scrna import PairedScrnaDataset
from models.scvi_ddpm_mlp_diffusion import ScviDdpmMlp
from trainers.scvi_ddpm_mlp_trainer import ScviDdpmMlpTrainer
from schedulers.warmup import GradualWarmupScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", default="configs/scvi_ddpm_mlp.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    # 1) 数据加载，返回 (x0, x1) 对
    dataset = PairedScrnaDataset(cfg.data.path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True  # Use pinned memory for faster GPU transfers
    )

    # 2) 模型
    model  = ScviDdpmMlp(cfg).to(device)

    # 3) 优化器 & Scheduler
    optim   = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    cosine  = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched   = GradualWarmupScheduler(optim,
                                     multiplier=cfg.train.warmup_multiplier,
                                     warm_epoch=cfg.train.epoch//10,
                                     after_scheduler=cosine)

    # 4) Trainer
    trainer = ScviDdpmMlpTrainer(model, optim, sched, loader, cfg)
    trainer.train()

if __name__=="__main__":
    main()
