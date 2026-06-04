"""Train drug-conditioned DDPM on MOA task (drug name + dose, or SMILES + dose)."""
import os
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.scrna import PairedScrnaDatasetDrugCond
from src.diffusion_baselines.models.scrna_ddpm_scrna import ScrnaDDPMDrugCond
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler
from src.diffusion_baselines.trainers.scrna_ddpm_scrna_trainer import ScRNATrainerDrugCond

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/baselines/scrna_ddpm_scrna.yaml")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-weight-dir", required=True)
    parser.add_argument("--gene-nums", type=int, default=None)
    parser.add_argument("--drug-key", default="perturbation")
    parser.add_argument("--dose-key", default="dose_value")
    parser.add_argument("--smiles-key", default="smiles")
    parser.add_argument("--use-drug-structure", action="store_true",
                        help="Use SMILES+dose Morgan fingerprint conditioning (Squidiff-style)")
    parser.add_argument("--drug-dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ckpt-save-interval", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.path = args.data_path
    cfg.train.save_weight_dir = args.save_weight_dir
    if args.gene_nums:
        cfg.model.input_dim = args.gene_nums
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.ckpt_save_interval is not None:
        cfg.train.ckpt_save_interval = args.ckpt_save_interval
    cfg.model.use_drug_structure = args.use_drug_structure
    cfg.model.drug_dimension = args.drug_dimension

    device = torch.device(cfg.train.device)
    dataset = PairedScrnaDatasetDrugCond(
        args.data_path,
        drug_key=args.drug_key,
        dose_key=args.dose_key,
        smiles_key=args.smiles_key,
        use_drug_structure=args.use_drug_structure,
        drug_dimension=args.drug_dimension,
    )
    if not args.use_drug_structure:
        num_drug = len(dataset.get_label_encoder().classes_)
        cfg.model.num_drug_classes = num_drug

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0,
    )

    model = ScrnaDDPMDrugCond(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, eps=1e-7)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=cfg.train.warmup_multiplier,
        warm_epoch=max(1, cfg.train.epoch // 10),
        after_scheduler=CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0.0))

    trainer = ScRNATrainerDrugCond(model, model.diffusion_trainer.to(device), optimizer, scheduler, loader, device, cfg)

    os.makedirs(args.save_weight_dir, exist_ok=True)
    meta_path = os.path.join(args.save_weight_dir, "label_encoder.npz")
    np.savez(
        meta_path,
        classes=dataset.get_label_encoder().classes_,
        use_drug_structure=np.array([args.use_drug_structure]),
        drug_dimension=np.array([args.drug_dimension]),
    )

    final_ckpt = os.path.join(args.save_weight_dir, "scrna_ddpm_epoch1000.pt")
    if os.path.exists(final_ckpt):
        print("Checkpoint exists, skipping.")
    else:
        trainer.train()

if __name__ == "__main__":
    main()
