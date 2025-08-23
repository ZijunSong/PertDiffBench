# models/scvi_ddpm_mlp_diffusion.py

import torch
import torch.nn as nn
import scanpy as sc
from .mlp_ddpm_mlp_diffusion import MLPDDPMMLP
from utils.scvi_utils import load_scvi_encoder
from anndata import AnnData

class ScviDdpmMlp(nn.Module):
    """
    Pipeline: scVI encoder(x0) → latent z0
              scVI encoder(x1) → latent z1 (train)
              DDPM(z0→z1) + MLP decoder
    """
    def __init__(self, cfg):
        super().__init__()
        # 1) 读取完成 setup_anndata 后的 AnnData 模板
        self.adata_template = sc.read_h5ad(
            cfg.model.scvi.get('adata_inference_path', cfg.data.path)
        )
        # 2) 加载 scVI 模型并获取原生推理函数
        raw_encode_fn, _ = load_scvi_encoder(
            model_dir=cfg.model.scvi.scvi_model_dir,
            adata=self.adata_template,
            device=cfg.train.device
        )
        # 3) 包装 encode_fn：Tensor → AnnData → latent Tensor
        def encode_tf(x: torch.Tensor) -> torch.Tensor:
            x_np = x.detach().cpu().numpy()
            ad = self.adata_template[: x_np.shape[0]].copy()
            ad.X = x_np
            z_np = raw_encode_fn(ad)  # [B, latent_dim]
            return torch.from_numpy(z_np).to(x.device)
        self.encode_fn = encode_tf

        # 4) 动态确定 scVI latent 维度
        # 使用 encode_fn 处理一个样本的模板表达
        sample_x = torch.from_numpy(
            self.adata_template.X[:1].A if hasattr(self.adata_template.X, 'A') else self.adata_template.X[:1]
        ).float()
        z_sample = self.encode_fn(sample_x)
        latent_dim = z_sample.shape[1]

        # 5) 覆写 AE 配置，使 MLPDDPMMLP 输入输出匹配 latent_dim
        gene_dim = self.adata_template.shape[1]
        # ae.input_dim 用于 ScRNADecoder 的 output_dim
        cfg.model.ae.input_dim = gene_dim
        cfg.model.ae.latent_dim = latent_dim
        # hidden_dim 不变
        # 构建 DDPM-MLP
        self.ddpm = MLPDDPMMLP(cfg)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        z0 = self.encode_fn(x0)
        z1 = self.encode_fn(x1)
        return self.ddpm.diffusion_trainer(z1, cond=z0)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor) -> torch.Tensor:
        z0 = self.encode_fn(x0)
        # 逆扩散采样
        T = self.ddpm.diffusion_sampler.T
        device = z0.device
        z_t = torch.randn_like(z0)
        B = z0.shape[0]
        for step in reversed(range(T)):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            eps  = self.ddpm.net(z_t, z0, t)
            mean = self.ddpm.diffusion_sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var  = self.ddpm.diffusion_sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
        x1 = self.ddpm.decoder(z_t)
        return x1.clamp(-1, 1)
