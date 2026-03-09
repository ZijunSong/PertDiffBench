# utils/scvi_utils.py

import scanpy as sc
import scvi

def load_scvi_encoder(model_dir, adata, device="gpu"):
    """
    Load a pretrained scvi-tools SCVI model and return its encoder/decoder.
    """
    vae = scvi.model.SCVI.load(model_dir, adata=adata)

    def encode_fn(x):
        """
        x: torch.Tensor [B, G], raw or normalized expression
        returns: Tensor [B, L]
        """
        # scvi-tools 1.x: use get_latent_representation
        return vae.get_latent_representation(x)

    def decode_fn(z):
        """
        z: torch.Tensor [B, L]
        returns: Tensor [B, G]
        """
        # Access the decoder module
        return vae.module.decoder(z)

    return encode_fn, decode_fn
