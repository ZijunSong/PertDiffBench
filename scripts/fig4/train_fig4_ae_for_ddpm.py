#!/usr/bin/env python3
"""
为 fig4 的 DDPM baseline 单独训练一个 VAE（仅 encoder + decoder，与 DDPM+MLP 同结构），
用于 4h/6h 的 2h/8h latent 线性插值生成。保存的 state_dict 仅含 encoder/decoder，
sample_fig4_vae_linear_interp.py 会以 strict=False 加载到 MLPDDPMMLP 中使用。
"""
import os
import sys
import argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.diffusion_baselines.models.mlp_ddpm_mlp_autoencoder import ScRNAEncoder, ScRNADecoder


class AEOnly(nn.Module):
    """仅 encoder + decoder，与 MLPDDPMMLP 的对应部分结构一致，便于 load_state_dict(..., strict=False)。"""
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = ScRNAEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = ScRNADecoder(latent_dim, input_dim, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z).clamp(-1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Train standalone AE on fig4 for DDPM 4h/6h linear interp")
    parser.add_argument("--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("--data-path", required=True, help="fig4_train.h5ad")
    parser.add_argument("--save-dir", required=True, help="e.g. checkpoints/fig4_ae_ddpm/fig4_3000/run1")
    parser.add_argument("--gene-nums", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    input_dim = args.gene_nums
    latent_dim = int(cfg.model.ae.latent_dim)
    hidden_dim = int(cfg.model.ae.hidden_dim)
    device = torch.device(cfg.train.device if hasattr(cfg.train, "device") else "cuda")

    adata = sc.read_h5ad(args.data_path)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # 与常见 pipeline 一致：clip 到 [-1,1]
    X = np.clip(X, -1.0, 1.0)
    X_t = torch.from_numpy(X)

    dataset = TensorDataset(X_t, X_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = AEOnly(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "ae_epoch_1000.pth")
    if os.path.isfile(ckpt_path):
        print(f"Found existing AE checkpoint: {ckpt_path}. Skipping training.")
        return

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n_batches,1):.6f}")

    # 保存为与 MLPDDPMMLP 兼容的 state_dict（仅 encoder/decoder）
    state = {"model_state_dict": model.state_dict()}
    torch.save(state, ckpt_path)
    print(f"Saved AE checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
