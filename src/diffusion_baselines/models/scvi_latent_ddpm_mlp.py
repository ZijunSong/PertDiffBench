import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        exp_term = -torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device))
        omega = torch.exp(
            exp_term * torch.arange(half_dim, device=device, dtype=torch.float32) / (half_dim - 1)
        )
        args = timesteps[:, None].float() * omega[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPCond(nn.Module):
    """
    Noise predictor ε_θ(z_t, cond=z0, t) in latent space.
    input: current z_t, cond z0, timestep t
    output: pred noise ε
    """
    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, time_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t = nn.Linear(time_dim, hidden_dim)

        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        target_dtype = self.fc1.weight.dtype
        z = z.to(dtype=target_dtype)
        cond = cond.to(dtype=target_dtype)

        # time embedding
        te = self.time_emb(t)
        te = self.act(self.fc_t(te))

        h = torch.cat([z, cond], dim=-1)
        h = self.act(self.fc1(h))
        h = h + te

        return self.fc2(h)


class LatentDecoderMLP(nn.Module):
    """
    simple MLP decoder: latent -> gene expression
    This decoder is supervised with (z1, x1), so x_hat ≈ x1.
    """
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        out_activation: str = "none",
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, gene_dim)
        self.out_activation = out_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z.float())
        x = self.out(h)
        if self.out_activation == "relu":
            x = torch.relu(x)
        elif self.out_activation == "softplus":
            x = torch.nn.functional.softplus(x)
        return x


class ScviLatentDDPMMLP(nn.Module):
    """
    Full architecture (encoder not included):
        z0, z1: from pretrained scVI latent
        net:    DDPM noise prednet
        decoder: MLP latent -> gene expression

    Training:
        - loss_diff in latent space (depends on z0, z1)
        - loss_dec in gene space (depends on z1, x1)
        - total loss = loss_diff + lambda_dec * loss_dec

    Sampling:
        - reverse diffuse from z0 (control latent) to get z_hat1
        - decode z_hat1 to gene expression with the decoder.
    """
    def __init__(self, cfg):
        super().__init__()

        ae_cfg = cfg.model.ae
        diff_cfg = cfg.model.diffusion

        self.latent_dim = ae_cfg.latent_dim   # == scVI n_latent
        self.gene_dim = ae_cfg.input_dim  # original gene count

        # decoder: latent -> gene
        self.decoder = LatentDecoderMLP(
            latent_dim=self.latent_dim,
            gene_dim=self.gene_dim,
            hidden_dim=ae_cfg.hidden_dim,
            n_layers=getattr(ae_cfg, "n_layers", 2),
            dropout=getattr(ae_cfg, "dropout", 0.1),
            use_layernorm=getattr(ae_cfg, "use_layernorm", True),
            out_activation=getattr(ae_cfg, "out_activation", "none"),
        )

        # DDPM noisenet
        self.net = MLPCond(
            latent_dim=self.latent_dim,
            hidden_dim=diff_cfg.hidden_dim,
            cond_dim=self.latent_dim,
            time_dim=diff_cfg.hidden_dim,
        )

        # Diffusion schedule
        beta1, betaT, T = diff_cfg.beta_1, diff_cfg.beta_T, diff_cfg.timesteps
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.net,
            beta_1=beta1,
            beta_T=betaT,
            T=T,
            conditional=True,
        )
        self.diffusion_sampler = GaussianDiffusionSampler(
            model=self.net,
            beta_1=beta1,
            beta_T=betaT,
            T=T,
        )

        # decoder loss weights
        self.lambda_dec = getattr(cfg.model, "lambda_dec", 1.0)

        self.mse = nn.MSELoss()

    def compute_loss(self, z0, z1, x1):
        """
        :param z0: scVI latent of control cells  [B, L]
        :param z1: scVI latent of perturbed cells [B, L]
        :param x1: gene expression of perturbed cells [B, G]
        """
        # 1) latent diffusion loss
        loss_diff = self.diffusion_trainer(z1, cond=z0)

        # 2) decoder reconstruction loss (z1 -> x1_hat ≈ x1)
        x1_hat = self.decoder(z1)
        loss_dec = self.mse(x1_hat, x1)

        loss_total = loss_diff + self.lambda_dec * loss_dec
        return loss_total, loss_diff, loss_dec

    @torch.no_grad()
    def sample_from_latent(self, z0: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Given control latent z0, reverse diffuse to z_hat1, then decode to gene expression.
        """
        device = z0.device
        if noise is None:
            z_t = torch.randn_like(z0)
        else:
            z_t = noise.to(device)

        B = z0.shape[0]
        z_clip = 1e3  # clip latent magnitude to avoid float32 overflow/NaN during reverse diffusion
        for step in reversed(range(self.diffusion_trainer.T)):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            eps = self.net(z_t, z0, t)
            mean = self.diffusion_sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.diffusion_sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
            z_t = torch.nan_to_num(z_t, nan=0.0, posinf=z_clip, neginf=-z_clip)
            z_t = z_t.clamp(min=-z_clip, max=z_clip)

        # decode latent to gene expression
        z_t = torch.nan_to_num(z_t, nan=0.0, posinf=z_clip, neginf=-z_clip)
        x_hat = self.decoder(z_t)
        return x_hat
