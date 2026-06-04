#!/usr/bin/env python
# -*- coding: utf-8 -*-

import anndata
import numpy as np
import scipy.sparse as sp
import os


def generate_lognormal_bionoise_files(file_path, output_dir):
    """
    Simulate cell-to-cell biological variation via log-normal + Poisson sampling.
    Build multiple perturbation datasets at different CV levels from an input h5ad.

    Biological noise simulator:
      1) For each gene, take its mean expression m_j across cells.
      2) For each target coefficient of variation (CV = c), assume:
            X_ij ~ LogNormal(mu_j(c), sigma(c)^2)
         where
            sigma(c)^2 = ln(c^2 + 1)
            mu_j(c)    = ln(m_j) - 0.5 * sigma(c)^2
      3) Then sample observed counts via:
            K_ij ~ Poisson(X_ij)

    This implements a BigSur-style model: log-normal latent rates + Poisson counts.
    """

    # 1. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Output directory ready: {output_dir} ---")

    # 2. Load original h5ad file
    print(f"--- Loading data: {file_path} ---")
    adata_original = anndata.read_h5ad(file_path)

    print("\n--- Original data layout ---")
    print(adata_original)

    is_sparse = sp.issparse(adata_original.X)
    print(f"\nexpression matrix (adata.X) is sparse: {is_sparse}")

    # Convert to dense array if necessary
    # NOTE: For very large data, this may be memory-heavy.
    original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()

    # 3. Compute per-gene means across cells as baseline expression
    print("\n--- Computing per-gene mean expression (log-normal target means) ---")
    gene_means = np.asarray(original_data.mean(axis=0)).ravel()

    # To avoid log(0), replace non-positive means with a small epsilon
    epsilon = 1e-3
    gene_means_safe = np.where(gene_means > 0, gene_means, epsilon)

    # 4. Define different CV levels for biological noise
    cv_levels = [0.1, 0.25, 0.5, 1.0, 1.5]
    print(f"\n--- Biological noise CV levels: {cv_levels} ---")

    n_cells, n_genes = original_data.shape
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # 5. Loop over CV levels and generate synthetic datasets
    for c in cv_levels:
        print(f"\n========== Processing biological noise level: CV = {c} ==========")

        sigma2 = np.log(c ** 2 + 1.0)
        sigma = np.sqrt(sigma2)
        mu = np.log(gene_means_safe) - 0.5 * sigma2

        print(f"For CV = {c}: sigma^2 = {sigma2:.6f}, sigma = {sigma:.6f}")
        print("mu uses per-gene ln(mean) shift (gene-specific mu_j)")

        mu_row = mu.reshape(1, -1)

        print("--- Sampling latent rates from log-normal (may be slow) ---")
        latent_rates = np.random.lognormal(mean=mu_row, sigma=sigma, size=(n_cells, n_genes))

        print("--- Sampling observed counts from Poisson given X ---")
        noisy_counts = np.random.poisson(lam=latent_rates).astype(np.float32)

        adata_noisy = anndata.AnnData(
            noisy_counts,
            obs=adata_original.obs.copy(),
            var=adata_original.var.copy()
        )

        adata_noisy.uns["bionoise_model"] = "LogNormal+Poisson"
        adata_noisy.uns["bionoise_cv"] = float(c)
        adata_noisy.uns["bionoise_sigma2"] = float(sigma2)
        adata_noisy.uns["bionoise_note"] = (
            "Counts generated via gene-wise LogNormal(mu_j(c), sigma(c)^2) "
            "followed by Poisson sampling, using original gene means as targets."
        )

        output_filename = f"{base_filename}_lognorm_cv_{c}.h5ad"
        output_path = os.path.join(output_dir, output_filename)
        print(f"Saving to: {output_path}")
        adata_noisy.write_h5ad(output_path)

    print("\n--- All log-normal biological noise datasets generated ---")


if __name__ == "__main__":
    input_file = "/share/PertBench/data/fig1/raw_task1/task1_train_CD4T_exp.h5ad"

    if not os.path.exists(input_file):
        print(f"Not found '{input_file}'. Creating dummy h5ad for testing...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.poisson(lam=5.0, size=(n_obs, n_vars)).astype(np.float32)
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f"Created dummy file '{input_file}'.")

    output_directory = "/share/PertBench/data/add_lognormal_bionoise_output"

    generate_lognormal_bionoise_files(input_file, output_directory)
