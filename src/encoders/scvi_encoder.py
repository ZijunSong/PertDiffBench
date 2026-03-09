# src/encoders/scvi_encoder.py

import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import scvi


class ScVIEncoderWrapper(nn.Module):
    """
    A thin PyTorch wrapper around a trained scvi.model.SCVI.

    It does NOT try to bypass scvi-tools internals.
    Instead, it uses the public API `get_latent_representation`
    with cell indices, which is the canonical way to get z.

    Typical usage:
        adata = sc.read_h5ad("data_with_scvi_setup.h5ad")
        model = scvi.model.SCVI.load("checkpoints/scvi_encoder", adata=adata)
        encoder = ScVIEncoderWrapper(model)

        # indices: 1D LongTensor or numpy array of cell indices
        z = encoder(indices)  # [B, n_latent]
    """

    def __init__(self, scvi_model: scvi.model.SCVI):
        super().__init__()
        # scvi_model is NOT a plain nn.Module, but we keep a reference.
        self.scvi_model = scvi_model
        # read latent dim from model
        self.n_latent = scvi_model.n_latent

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        :param indices: 1D tensor of cell indices into the AnnData
                        that was used to train the SCVI model.
                        Shape: [B]
        :return: latent embedding z, shape [B, n_latent], as torch.float32
        """
        if indices.ndim != 1:
            raise ValueError(
                f"indices should be 1D, got shape {indices.shape}"
            )

        # Move indices to CPU and convert to numpy for scvi
        idx_np = indices.detach().cpu().numpy().astype(np.int64)

        # scvi handles batching internally with batch_size argument
        latent = self.scvi_model.get_latent_representation(
            indices=idx_np,
            batch_size=len(idx_np),
        )
        # latent is numpy array [B, n_latent]
        z = torch.from_numpy(latent).float()

        # Put result on same device as indices for convenience
        return z.to(indices.device)


def load_scvi_encoder(model_dir: str, adata_path: str, device: str = "cpu") -> ScVIEncoderWrapper:
    """
    Convenience helper to load a trained SCVI model and wrap it as encoder.

    :param model_dir: Directory where SCVI was saved via model.save(...).
    :param adata_path: Path to the AnnData file used for training (or a compatible one).
    :param device: "cpu" or "cuda".
    :return: ScVIEncoderWrapper instance.
    """
    adata = sc.read_h5ad(adata_path)
    # Ensure AnnData is registered the same way as during training.
    # If you saved the model with adata inside, you can also omit adata here.
    scvi_model = scvi.model.SCVI.load(model_dir, adata=adata)
    scvi_model.to_device(device)
    scvi_model.eval()

    encoder = ScVIEncoderWrapper(scvi_model)
    return encoder
