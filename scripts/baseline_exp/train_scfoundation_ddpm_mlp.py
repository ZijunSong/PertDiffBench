# scripts/baaseline/train_scfoundation_ddpm_mlp.py

import argparse
from omegaconf import OmegaConf

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import the new, simple dataset
from data.scrna import EmbeddingDataset 
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler
# Import the new, simplified model and trainer
from src.diffusion_baselines.models.scfoundation_ddpm_mlp_diffusion import LatentDiffusionModel
from src.diffusion_baselines.trainers.scfoundation_ddpm_mlp_trainer import LatentDDPMTrainer

def main():
    p = argparse.ArgumentParser()
    # You might want to create a new, simpler yaml config file
    p.add_argument("-c","--config",default="configs/baselines/scfoundation_ddpm_mlp.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    # Use the new EmbeddingDataset
    # The path to the .npy file should be in your config
    dataset = EmbeddingDataset(cfg.data.embedding_path) 
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    # Instantiate the new simplified model
    model = LatentDiffusionModel(cfg).to(device)

    optim  = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched  = GradualWarmupScheduler(optim, multiplier=cfg.train.warmup_multiplier,
                                     warm_epoch=cfg.train.epoch//10, after_scheduler=cosine)

    # Instantiate the new trainer
    trainer = LatentDDPMTrainer(model, optim, sched, loader, cfg)
    trainer.train()

if __name__=="__main__":
    main()