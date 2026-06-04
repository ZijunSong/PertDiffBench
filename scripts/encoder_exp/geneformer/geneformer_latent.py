#!/usr/bin/env python3
# scripts/encoder_exp/geneformer_latent.py

import anndata as ad
import torch
import numpy as np
from torch.utils.data import Dataset


class PairedGeneformerLatentDataset(Dataset):
    """
    Paired dataset in Geneformer latent space, vs handle .

     : 
      - using obs['condition'] ctrl vs treated.
      - based on condition control label ('ctrl', 'control', 'unperturbed', 'DMSO' ).
      - no condition cols, "before ctrl, after treated" .

    no longer requires 'is_control' 'pair_id' cols.
    """

    def __init__(self, h5ad_path: str, split: str = "train",
                 cond_key: str = "condition", ctrl_label: str | None = None):
        super().__init__()
        print(f"[PairedGeneformerLatentDataset] Loading {h5ad_path} (split={split})")
        self.adata = ad.read_h5ad(h5ad_path)

        if "X_geneformer" not in self.adata.obsm.keys():
            raise KeyError("obsm['X_geneformer'] not found. Did you run Geneformer encoder?")

        obs = self.adata.obs

        # --------- 1. split filter ( exist ) ----------
        if "split" in obs.columns:
            mask = (obs["split"] == split).to_numpy()
            print(f"[PairedGeneformerLatentDataset] Using obs['split'] == '{split}', "
                  f"selected {mask.sum()}/{len(mask)} cells.")
        elif "set" in obs.columns:
            mask = (obs["set"] == split).to_numpy()
            print(f"[PairedGeneformerLatentDataset] Using obs['set'] == '{split}', "
                  f"selected {mask.sum()}/{len(mask)} cells.")
        else:
            mask = np.ones(self.adata.n_obs, dtype=bool)
            print(f"[PairedGeneformerLatentDataset] No 'split' or 'set' column found; using ALL {len(mask)} cells.")

        mask = np.asarray(mask, dtype=bool)

        latent_all = self.adata.obsm["X_geneformer"]
        if hasattr(latent_all, "toarray"):
            latent_all = latent_all.toarray()
        self.latent = latent_all[mask, :]

        self.obs = obs[mask].copy()

        # --------- 2. based on condition build ctrl / treated for ----------
        if cond_key in self.obs.columns:
            cond = self.obs[cond_key].astype(str)
            uniq = cond.unique().tolist()
            print(f"[PairedGeneformerLatentDataset] Found obs['{cond_key}'] with {len(uniq)} unique values.")

            # control label
            if ctrl_label is None:
                candidates = ["ctrl", "control", "unperturbed", "DMSO", "dmso", "vehicle"]
                ctrl_label = None
                for cand in candidates:
                    if cand in uniq:
                        ctrl_label = cand
                        break
                if ctrl_label is None:
                    # : using count condition as control
                    value_counts = cond.value_counts()
                    ctrl_label = value_counts.index[0]
                print(f"[PairedGeneformerLatentDataset] Inferred ctrl_label = '{ctrl_label}'")

            is_ctrl = (cond == ctrl_label).to_numpy()
            ctrl_idx = np.where(is_ctrl)[0]
            trt_idx = np.where(~is_ctrl)[0]

            if len(ctrl_idx) == 0 or len(trt_idx) == 0:
                print("[PairedGeneformerLatentDataset] Warning: no clear ctrl/treated split from condition; "
                      "falling back to simple half-half split.")
                self._build_pairs_half_half()
            else:
                n_pairs = min(len(ctrl_idx), len(trt_idx))
                ctrl_idx = ctrl_idx[:n_pairs]
                trt_idx = trt_idx[:n_pairs]
                self.pairs = [(int(i_c), int(i_t)) for i_c, i_t in zip(ctrl_idx, trt_idx)]
                print(f"[PairedGeneformerLatentDataset] Built {len(self.pairs)} pairs "
                      f"(ctrl={len(ctrl_idx)}, treated={len(trt_idx)}).")
        else:
            print(f"[PairedGeneformerLatentDataset] obs['{cond_key}'] not found; "
                  f"falling back to simple half-half split.")
            self._build_pairs_half_half()

    def _build_pairs_half_half(self):
        """ no condition batch info, before ctrl, after treated."""
        n = self.latent.shape[0]
        if n < 2:
            raise ValueError("Not enough cells to build pairs.")
        half = n // 2
        ctrl_idx = np.arange(0, half)
        trt_idx = np.arange(half, half + len(ctrl_idx))
        n_pairs = min(len(ctrl_idx), len(trt_idx))
        ctrl_idx = ctrl_idx[:n_pairs]
        trt_idx = trt_idx[:n_pairs]
        self.pairs = [(int(i_c), int(i_t)) for i_c, i_t in zip(ctrl_idx, trt_idx)]
        print(f"[PairedGeneformerLatentDataset] Half-half pairing: {len(self.pairs)} pairs "
              f"(ctrl={len(ctrl_idx)}, treated={len(trt_idx)}).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i_ctr, i_trt = self.pairs[idx]
        x0 = self.latent[i_ctr].astype("float32")
        x1 = self.latent[i_trt].astype("float32")
        return torch.from_numpy(x0), torch.from_numpy(x1)
