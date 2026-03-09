"""Train drug-conditioned MLP-DDPM-MLP on MOA task."""
import os
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.scrna import PairedScrnaDatasetDrugCond
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLPDrugCond
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler
from src.diffusion_baselines.trainers.mlp_ddpm_mlp_trainer import ScRNATrainerDrugCond

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-weight-dir", required=True)
    parser.add_argument("--gene-nums", type=int, default=None)
    parser.add_argument("--drug-key", default="perturbation")
    parser.add_argument("--dose-key", default="dose_value")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.path = args.data_path
    cfg.train.save_weight_dir = args.save_weight_dir
    if args.gene_nums:
        cfg.model.ae.input_dim = args.gene_nums

    device = torch.device(cfg.train.device)
    dataset = PairedScrnaDatasetDrugCond(args.data_path, drug_key=args.drug_key, dose_key=args.dose_key)
    num_drug = len(dataset.get_label_encoder().classes_)
    cfg.model.num_drug_classes = num_drug

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers, pin_memory=True)

    model = MLPDDPMMLPDrugCond(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=cfg.train.warmup_multiplier,
        warm_epoch=cfg.train.epoch // 10,
        after_scheduler=CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0))

    trainer = ScRNATrainerDrugCond(model, model.diffusion_trainer.to(device), optimizer, scheduler, loader, device, cfg)

    os.makedirs(args.save_weight_dir, exist_ok=True)
    np.savez(os.path.join(args.save_weight_dir, "label_encoder.npz"), classes=dataset.get_label_encoder().classes_)

    final_ckpt = os.path.join(args.save_weight_dir, "model_epoch_1000.pth")
    if os.path.exists(final_ckpt):
        print("Checkpoint exists, skipping.")
    else:
        trainer.train()

if __name__ == "__main__":
    main()
