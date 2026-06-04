# models/mlp_ddpm_mlp_predictor.py

import torch
import torch.nn as nn

class MLPNoisePredictor(nn.Module):
    """
    εθ(z_t, t | cond) for latent vectors z_t, with conditioning vector cond.
    Args:
        latent_dim (int): z_t dimension
        hidden_dim (int): hidden layer dimension
        cond_dim (int): conditioning vector dimension
    """
    def __init__(self, latent_dim, hidden_dim, cond_dim):
        super().__init__()
        # inputdim   = z_t + cond
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
        :param z: current noisy latent, shape [B, latent_dim]
        :param t: timestep (currently unused)
        :param cond: conditioning latent, shape [B, cond_dim]
        :return:     pred  noise, shape [B, latent_dim]
        """
        # concat z_t and cond
        x = torch.cat([z, cond], dim=-1)
        return self.net(x)
