#!/usr/bin/env python3
# scripts/train_mlp_ddpm_mlp.py

import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_files.scrna import PairedScrnaDataset
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP
from src.diffusion_baselines.trainers.mlp_ddpm_mlp_trainer import ScRNATrainer
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/mlp_ddpm_mlp.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-weight-dir", type=str, default=None,
                        help="Override train.save_weight_dir in the YAML config")
    parser.add_argument("--gene-nums", type=int, default=None,
                        help="Override model.ae.input_dim in the YAML config")
    parser.add_argument("--pair-only-obs-key", type=str, default=None,
                        help="scGen-style: build pairs only where this obs column equals pair-only-obs-value")
    parser.add_argument("--pair-only-obs-value", type=str, default=None,
                        help="scGen-style: use with pair-only-obs-key (e.g. 'train')")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.data_path:
        print(f"Overriding data.path from command line: '{cfg.data.path}' -> '{args.data_path}'")
        cfg.data.path = args.data_path
    
    if args.save_weight_dir:
        print(f"Overriding train.save_weight_dir from command line: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'")
        cfg.train.save_weight_dir = args.save_weight_dir

    if args.gene_nums:
        print(f"Overriding model.ae.input_dim from command line: '{cfg.model.ae.input_dim}' -> '{args.gene_nums}'")
        cfg.model.ae.input_dim = args.gene_nums

    device = torch.device(cfg.train.device)

    print("Loading H5AD dataset from:", os.path.abspath(cfg.data.path))
    pair_kw = {}
    if args.pair_only_obs_key is not None and args.pair_only_obs_value is not None:
        pair_kw = {"pair_only_obs_key": args.pair_only_obs_key, "pair_only_obs_value": args.pair_only_obs_value}
    dataset = PairedScrnaDataset(cfg.data.path, **pair_kw)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True  # Use pinned memory for faster GPU transfers
    )
    model = MLPDDPMMLP(cfg).to(device)

    optim = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched  = GradualWarmupScheduler(optim, multiplier=cfg.train.warmup_multiplier,
                                    warm_epoch=cfg.train.epoch//10, after_scheduler=cosine)

    trainer = ScRNATrainer(model, model.diffusion_trainer.to(device), optim, sched, loader, device, cfg)
    
    final_model_path = os.path.join(cfg.train.save_weight_dir, 'model_epoch_1000.pth')

    if os.path.exists(final_model_path):
        print(f"Found pre-trained model at '{final_model_path}'. Skipping training.")
    else:
        print("No pre-trained model found. Starting training...")
        trainer.train()

if __name__=="__main__":
    main()
