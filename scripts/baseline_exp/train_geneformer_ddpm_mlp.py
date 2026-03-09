#!/usr/bin/env python3
import os, argparse
from omegaconf import OmegaConf

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.scrna import PairedScrnaDataset
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler
# MODIFIED: Import the new GeneFormer-based model
from src.diffusion_baselines.models.geneformer_ddpm_mlp_diffusion import MLPDDPMMLPGeneFormer
# MODIFIED: Import the new trainer class
from src.diffusion_baselines.trainers.geneformer_ddpm_mlp_trainer import MLPDDPMMLPGeneFormerTrainer

def main():
    p = argparse.ArgumentParser()
    # MODIFIED: Point to the new config file by default
    p.add_argument("-c", "--config", default="configs/baseline/geneformer_ddpm_mlp.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    dataset = PairedScrnaDataset(cfg.data.path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    # MODIFIED: Instantiate the GeneFormer-based model
    model = MLPDDPMMLPGeneFormer(cfg).to(device)

    optim = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched = GradualWarmupScheduler(optim, multiplier=cfg.train.warmup_multiplier,
                                   warm_epoch=cfg.train.epoch//10, after_scheduler=cosine)

    # MODIFIED: Instantiate the new trainer
    trainer = MLPDDPMMLPGeneFormerTrainer(model, optim, sched, loader, cfg)
    trainer.train()

if __name__=="__main__":
    main()