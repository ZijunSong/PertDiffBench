"""
Train a conditional DDPM (Denoising Diffusion Probabilistic Model) on paired
scRNA-seq data (Control → IFN conditions) using PyTorch.

This script performs the following steps:
1. Parse command-line arguments.
2. Load training configuration from YAML.
3. Initialize device, dataset, and DataLoader.
4. Build the conditional DDPM model.
5. Configure optimizer and learning rate schedulers.
6. Instantiate the ScRNATrainer and run the training loop.
"""

import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.scrna import PairedScrnaDataset
from src.diffusion_baselines.models.scrna_ddpm_scrna import ScrnaDDPM
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler
from src.diffusion_baselines.trainers.scrna_ddpm_scrna_trainer import ScRNATrainer



def main():
    """
    Main entrypoint for training the conditional DDPM on scRNA-seq data.
    Parses arguments, sets up training components, and launches the trainer.
    """
    # ------------------------------------------------------------------------------
    # 1) Argument parsing
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train DDPM on paired scRNA-seq data (Control → IFN)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/scrna_ddpm_scrna.yaml",
        help="Path to the DDPM training YAML configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None, 
        help="Override data.path in YAML"
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override train.save_weight_dir in YAML"
    )
    parser.add_argument(
        "--gene-nums",
        type=int,
        default=None,
        help="Override model.input_dim in YAML"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # 2) Load configuration and set device
    # ------------------------------------------------------------------------------
    cfg = OmegaConf.load(args.config)

    if args.data_path:
        print(f"Overriding data.path from command line: '{cfg.data.path}' -> '{args.data_path}'")
        cfg.data.path = args.data_path
    
    if args.save_weight_dir:
        print(f"Overriding train.save_weight_dir from command line: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'")
        cfg.train.save_weight_dir = args.save_weight_dir

    if args.gene_nums:
        print(f"Overriding model.input_dim from command line: '{cfg.model.input_dim}' -> '{args.gene_nums}'")
        cfg.model.input_dim = args.gene_nums

    device = torch.device(cfg.train.device)  # e.g., "cuda" or "cpu"

    # ------------------------------------------------------------------------------
    # 3) Prepare dataset and DataLoader
    # ------------------------------------------------------------------------------
    # Log the absolute path for reproducibility
    print("Loading H5AD dataset from:", os.path.abspath(cfg.data.path))
    dataset = PairedScrnaDataset(cfg.data.path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True  # Use pinned memory for faster GPU transfers
    )

    # ------------------------------------------------------------------------------
    # 4) Build the conditional DDPM model
    # ------------------------------------------------------------------------------
    model = ScrnaDDPM(cfg).to(device)

    # ------------------------------------------------------------------------------
    # 5) Configure optimizer and learning rate schedulers
    # ------------------------------------------------------------------------------
    # Use AdamW optimizer for weight decay decoupling
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        eps=1e-7
    )
    # Cosine annealing scheduler for cyclic LR decay
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.train.epoch,
        eta_min=0.0
    )
    # Gradual warm-up scheduler before cosine decay
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=max(1, cfg.train.epoch // 10),
        after_scheduler=cosine_scheduler
    )

    # ------------------------------------------------------------------------------
    # 6) Instantiate trainer and launch training
    # ------------------------------------------------------------------------------
    trainer = ScRNATrainer(
        model=model,                                      # DDPM model instance
        diffusion_trainer=model.diffusion_trainer.to(device),  # Underlying diffusion trainer
        optimizer=optimizer,                                  # Optimizer for model parameters
        scheduler=scheduler,                                  # LR scheduler with warmup
        loader=loader,                                        # DataLoader for paired scRNA batches
        device=device,                                        # Device to run training on
        cfg=cfg                                               # Full configuration object
    )

    final_model_path = os.path.join(cfg.train.save_weight_dir, 'scrna_ddpm_epoch1000.pt')

    if os.path.exists(final_model_path):
        print(f"Found pre-trained model at '{final_model_path}'. Skipping training.")
    else:
        print("No pre-trained model found. Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
