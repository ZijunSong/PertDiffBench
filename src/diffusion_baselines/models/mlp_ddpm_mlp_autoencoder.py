# models/mlp_ddpm_mlp_autoencoder.py

import torch.nn as nn

class ScRNAEncoder(nn.Module):
    """MLP encoder: [B, G] → [B, L]"""
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class ScRNADecoder(nn.Module):
    """MLP decoder: [B, L] → [B, G]"""
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__":
    # Sanity check
    encoder = ScRNAEncoder(input_dim=2000, latent_dim=64, hidden_dim=512)
    decoder = ScRNADecoder(latent_dim=64, output_dim=2000, hidden_dim=512)

    import torch
    x = torch.randn(8, 2000)
    z = encoder(x)
    x_recon = decoder(z)

    print("Input shape:", x.shape)
    print("Latent shape:", z.shape)
    print("Reconstructed shape:", x_recon.shape)