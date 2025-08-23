# models/scrna_ddpm_scrna.py

"""
Conditional DDPM model for paired scRNA-seq data (Control → IFN).
Implements noise scheduling, model definition, and conditional sampling.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

def get_noise_schedule(cfg):
    """
    Extract the diffusion noise schedule parameters from the config.

    Args:
        cfg: OmegaConf configuration object with a 'diffusion' section.

    Returns:
        Tuple of (beta_start, beta_end, num_timesteps).
    """
    beta_1 = cfg.diffusion.beta_1
    beta_T = cfg.diffusion.beta_T
    T      = cfg.diffusion.timesteps
    return beta_1, beta_T, T

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for timesteps t.
    This follows the scheme from "Attention Is All You Need".
    """
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of the output embedding (should be even).
        """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute the sinusoidal embedding for each timestep.

        Args:
            timesteps: LongTensor of shape [B] containing timestep indices.

        Returns:
            FloatTensor of shape [B, dim] with concatenated sin and cos embeddings.
        """
        half = self.dim // 2
        # Compute the scaling factor: 10000^(2i/dim)
        device = timesteps.device
        # ensure float32 constants
        inv_freq = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device))
            * torch.arange(half, device=device, dtype=torch.float32)
            / (half - 1)
        )
        args = timesteps[:, None].float() * inv_freq[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class MLPCond(nn.Module):
    """
    A simple MLP that conditions on the noisy input and control vector,
    and incorporates timestep embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Number of features (genes) in the input vector.
            hidden_dim: Dimension of hidden layers and timestep embedding.
        """
        super().__init__()
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.fc_t     = nn.Linear(hidden_dim, hidden_dim)
        self.fc1      = nn.Linear(input_dim * 2, hidden_dim)
        self.act      = nn.SiLU()     # SiLU activation (Swish)
        self.fc2      = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the conditional MLP.

        Args:
            x: Noisy input tensor shape [B, D].
            cond: Control (clean) input tensor shape [B, D].
            t: Integer timesteps tensor shape [B].

        Returns:
            Predicted noise tensor shape [B, D].
        """
        # cast inputs to match weight dtype (float32)
        target_dtype = self.fc1.weight.dtype
        x    = x.to(dtype=target_dtype)
        cond = cond.to(dtype=target_dtype)

        # Compute timestep embeddings and project
        te = self.time_emb(t)         # [B, H]
        te = self.act(self.fc_t(te))  # [B, H]

        # Concatenate noisy input and condition vector
        h = torch.cat([x, cond], dim=1)    # [B, 2D]
        h = self.act(self.fc1(h)) + te     # Fuse with time embedding
        return self.fc2(h)                 # Predict noise ε

class ScrnaDDPM(nn.Module):
    """
    Conditional DDPM model wrapping the noise predictor and diffusion logic.
    """
    def __init__(self, cfg):
        """
        Initialize the DDPM model and associated trainer/sampler.

        Args:
            cfg: OmegaConf configuration containing model & diffusion settings.
        """
        super().__init__()
        D = cfg.model.input_dim
        H = cfg.model.hidden_dim

        # Instantiate the conditional noise predictor network
        self.net = MLPCond(D, H)

        # Extract noise schedule parameters
        beta_1, beta_T, T = get_noise_schedule(cfg)
        # Initialize the diffusion trainer (handles q-sampling & loss)
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.net, beta_1=beta_1, beta_T=beta_T, T=T, conditional=True
        )
        # Initialize the diffusion sampler (handles p-sampling)
        self.diffusion_sampler = GaussianDiffusionSampler(
            model=self.net, beta_1=beta_1, beta_T=beta_T, T=T
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Standard forward interface for noise prediction.

        Args:
            x_t: Noisy data tensor [B, D].
            cond: Control data tensor [B, D].
            t: LongTensor of timesteps [B].

        Returns:
            Predicted noise tensor [B, D].
        """
        return self.net(x_t, cond, t)

    @torch.no_grad()
    def sample_cond(self, cond: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Perform conditional reverse diffusion sampling.

        Args:
            cond: Control data tensor [B, D].
            noise: Optional initial noise [B, D]. If None, random noise is used.

        Returns:
            Denoised output tensor (predicted IFN) [B, D].
        """
        device = cond.device
        B, D = cond.shape

        # Initialize noise if not provided
        cond = cond.to(dtype=self.net.fc1.weight.dtype)
        x_t = noise.to(device).to(dtype=cond.dtype) if noise is not None else torch.randn_like(cond)

        # Loop from T-1 down to 0
        T = self.diffusion_trainer.T
        for timestep in reversed(range(T)):
            # Create a batch of the current timestep index
            t_batch = torch.full((B,), timestep, dtype=torch.long, device=device)
            # Predict noise εθ(x_t, cond, t)
            eps = self.net(x_t, cond, t_batch)
            # Compute mean of p(x_{t-1} | x_t)
            mean = self.diffusion_sampler.predict_xt_prev_mean_from_eps(x_t, t_batch, eps)
            # Get the posterior variance for this timestep
            var = self.diffusion_sampler.posterior_var[timestep]

            # Sample from the Gaussian at all but the final step
            if timestep > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean

        # Clamp to original data range
        return x_t.clamp(-1, 1)
