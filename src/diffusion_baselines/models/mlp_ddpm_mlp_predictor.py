# models/mlp_ddpm_mlp_predictor.py

import torch
import torch.nn as nn

class MLPNoisePredictor(nn.Module):
    """
    εθ(z_t, t | cond) for latent vectors z_t，带条件向量 cond。
    Args:
        latent_dim (int): z_t 的维度
        hidden_dim (int): 隐藏层维度
        cond_dim   (int): 条件向量 cond 的维度
    """
    def __init__(self, latent_dim, hidden_dim, cond_dim):
        super().__init__()
        # 输入维度 = z_t + cond
        input_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, t, cond):
        """
        :param z:    当前噪声 latent 张量，形状 [B, latent_dim]
        :param t:    时间步（目前未用到）
        :param cond: 条件 latent 张量，形状 [B, cond_dim]
        :return:     预测的噪声，形状 [B, latent_dim]
        """
        # 拼接 z_t 和 cond
        x = torch.cat([z, cond], dim=-1)
        return self.net(x)
