#!/usr/bin/env python3
import os, argparse
from omegaconf import OmegaConf

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.scrna import PairedScrnaDataset
from schedulers.warmup import GradualWarmupScheduler
from models.scgpt_ddpm_mlp_diffusion import MLPDDPMMLPscGPT
from trainers.scgpt_ddpm_mlp_trainer import MLPDDPMMLPscGPTTrainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config",default="configs/scgpt_ddpm_mlp.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    dataset = PairedScrnaDataset(cfg.data.path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True  # Use pinned memory for faster GPU transfers
    )
    model  = MLPDDPMMLPscGPT(cfg).to(device)

    optim  = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched  = GradualWarmupScheduler(optim, multiplier=cfg.train.warmup_multiplier,
                                     warm_epoch=cfg.train.epoch//10, after_scheduler=cosine)

    trainer = MLPDDPMMLPscGPTTrainer(model, optim, sched, loader, cfg)
    trainer.train()

if __name__=="__main__":
    main()
