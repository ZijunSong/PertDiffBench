#!/usr/bin/env python3
# scripts/train_scimilarity_ddpm_mlp.py

import os
import argparse

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.scrna import PairedScrnaDataset
from src.diffusion_baselines.models.scimilarity_ddpm_mlp_diffusion import (
    SCimilarityDDPMMLP,
)
from src.diffusion_baselines.trainers.mlp_ddpm_mlp_trainer import ScRNATrainer
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/scimilarity_ddpm_mlp.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data.path in the config.",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override train.save_weight_dir in the config.",
    )
    parser.add_argument(
        "--gene-nums",
        type=int,
        default=None,
        help="Override model.ae.input_dim in the config.",
    )
    parser.add_argument(
        "--scimilarity-model-path",
        type=str,
        default=None,
        help="Override model.scimilarity.model_path in the config.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Command line overrides
    if args.data_path:
        print(
            f"Overriding data.path from command line: "
            f"'{cfg.data.path}' -> '{args.data_path}'"
        )
        cfg.data.path = args.data_path

    if args.save_weight_dir:
        print(
            f"Overriding train.save_weight_dir from command line: "
            f"'{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'"
        )
        cfg.train.save_weight_dir = args.save_weight_dir

    if args.gene_nums:
        print(
            f"Overriding model.ae.input_dim from command line: "
            f"'{cfg.model.ae.input_dim}' -> '{args.gene_nums}'"
        )
        cfg.model.ae.input_dim = args.gene_nums

    if args.scimilarity_model_path:
        print(
            f"Overriding model.scimilarity.model_path from command line: "
            f"'{cfg.model.scimilarity.model_path}' "
            f"-> '{args.scimilarity_model_path}'"
        )
        cfg.model.scimilarity.model_path = args.scimilarity_model_path

    device = torch.device(cfg.train.device)

    print("Loading H5AD dataset from:", os.path.abspath(cfg.data.path))
    dataset = PairedScrnaDataset(cfg.data.path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # Build model with SCimilarity encoder
    model = SCimilarityDDPMMLP(cfg).to(device)

    # Optimizer & LR scheduler (same as original MLP baseline)
    optim = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    cosine = CosineAnnealingLR(
        optim, T_max=cfg.train.epoch, eta_min=0.0
    )
    sched = GradualWarmupScheduler(
        optim,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=cfg.train.epoch // 10,
        after_scheduler=cosine,
    )

    trainer = ScRNATrainer(
        model=model,
        diffusion=model.diffusion_trainer.to(device),
        optimizer=optim,
        scheduler=sched,
        data_loader=loader,
        device=device,
        cfg=cfg,
    )

    final_model_path = os.path.join(
        cfg.train.save_weight_dir, "model_epoch_1000.pth"
    )

    if os.path.exists(final_model_path):
        print(
            f"Found pre-trained model at '{final_model_path}'. Skipping training."
        )
    else:
        print("No pre-trained model found. Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
