# models/scgpt_ddpm_mlp_diffusion.py

import numpy as np
import scanpy as sc
import torch
from torch import nn
from .base import DiffusionModel
from utils.scgpt_utils import load_scgpt, embed_cells
from .mlp_ddpm_mlp_autoencoder import ScRNADecoder
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .mlp_ddpm_mlp_diffusion import SinusoidalPosEmb

class TimeConditionalWrapper(nn.Module):
    """为 core_net 添加时序步长嵌入，使其 signature 为 forward(z, t)"""
    def __init__(self, core_net: nn.Module, time_dim: int, latent_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t = nn.Linear(time_dim, latent_dim)
        self.net = core_net

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1) 构造时序嵌入
        te = self.time_emb(t)      # [B, time_dim]
        te = self.fc_t(te)         # [B, latent_dim]
        # 2) 把时序嵌入加到 z 上，再丢给 core_net
        return self.net(z + te)


class MLPDDPMMLPscGPT(DiffusionModel):
    def __init__(self, cfg):
        T     = cfg.model.diffusion.T
        betas = torch.linspace(cfg.model.diffusion.beta_1,
                               cfg.model.diffusion.beta_T,
                               T)
        super().__init__(T, betas)

        # 1) scGPT 加载
        device = torch.device(cfg.train.device)
        self.scgpt_model, self.scgpt_tokenizer = load_scgpt(cfg, device)

        # —— 在这里插入 hook 注册 —— 
        def stats_hook(name):
            def hook(module, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else out[0]
                if torch.isnan(tensor).any():
                    print(f"[NAN HUNT] NaN 出现在 {name} 的输出！ min/max = {tensor.min().item()}/{tensor.max().item()}")
                else:
                    print(f"[OK] {name} 输出正常， min/max = {tensor.min().item()}/{tensor.max().item()}")
            return hook

        for n, m in self.scgpt_model.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.GELU, torch.nn.Linear)):
                m.register_forward_hook(stats_hook(n))
        # —— hook 注册完毕 —— 
        
        adata_ref = sc.read_h5ad(cfg.data.path)
        self.gene_ids = adata_ref.var_names.to_list()

        # 2) 构建 core_net：只重建 latent_dim
        latent_dim = cfg.model.decoder.latent_dim
        hidden_dim = cfg.model.decoder.hidden_dim
        core_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3) 包装一个支持 time-step 的 wrapper
        time_dim = cfg.model.diffusion.hidden_dim
        conditioned_net = TimeConditionalWrapper(core_net, time_dim, latent_dim)

        # 4) Trainer & Sampler (无条件流程也会调用 conditioned_net(z, t))
        self.trainer = GaussianDiffusionTrainer(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T,
            conditional=False
        )
        self.sampler = GaussianDiffusionSampler(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T
        )

        # 5) 解码器
        self.decoder = ScRNADecoder(
            latent_dim,
            cfg.model.decoder.output_dim,
            hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 用 scGPT 得到 z0
        z0 = embed_cells(
            self.scgpt_model, self.scgpt_tokenizer,
            x, self.gene_ids
        ).float()  # [B, latent_dim]
        # 2) 构造全零的 time-step
        B = z0.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=z0.device)
        # 3) 交给 trainer
        return self.trainer(z0, t)

    @torch.no_grad()
    def sample(self, adata_ref):
        # 1) 同步获取 control embedding
        z0 = embed_cells(
            self.scgpt_model, self.scgpt_tokenizer,
            adata_ref.X, self.gene_ids
        ).float().to(self.betas.device)
        # 2) 初始化噪声
        z_t = torch.randn_like(z0)
        B = z0.shape[0]
        # 3) 迭代采样
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, dtype=torch.long, device=z0.device)
            eps = self.trainer.model(z_t, t)
            mean = self.sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
        # 4) 解码
        return self.decoder(z_t).clamp(-1, 1)
