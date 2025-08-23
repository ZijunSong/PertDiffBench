#!/usr/bin/env python3
# scripts/eval_ddpm.py

import os
import argparse
from omegaconf import OmegaConf

import torch
from torchvision.utils import save_image

from models.ddpm import DDPM
from models.gaussian_diffusion import GaussianDiffusionSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/ddpm_default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.train.device)

    # 1) 重建模型结构
    model = DDPM(cfg).to(device)

    # 2) 加载 checkpoint，并只取 model_state
    ckpt_path = os.path.join(
        cfg.train.save_weight_dir,
        f"ckpt_{cfg.train.epoch}.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    # 注意这里取 "model_state"
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) 构造 sampler
    sampler = GaussianDiffusionSampler(
        model.unet, 
        cfg.model.beta_1,
        cfg.model.beta_T,
        cfg.model.unet.T
    ).to(device)

    # 4) 从噪声采样
    n = cfg.sample.batch_size
    noise = torch.randn(n, 3, 32, 32, device=device)
    with torch.no_grad():
        samples = sampler(noise)
    samples = samples * 0.5 + 0.5

    # 5) 保存图片
    os.makedirs(cfg.sample.sampled_dir, exist_ok=True)
    for idx, img in enumerate(samples):
        # img 是 [3, H, W]，在 [0,1] 区间
        save_image(
            img,
            os.path.join(cfg.sample.sampled_dir, f"sample_{idx:03d}.png")
        )
    print(f"Saved {len(samples)} individual images to {cfg.sample.sampled_dir}")

if __name__ == "__main__":
    main()
