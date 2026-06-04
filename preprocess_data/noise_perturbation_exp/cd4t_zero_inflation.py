#!/usr/bin/env python
# -*- coding: utf-8 -*-

import anndata
import numpy as np
import scipy.sparse as sp
import os


def generate_zeroinflated_dropout_files(file_path, output_dir):
    """
    Simulate technical dropout via gene-specific zero-inflation.

    Zero-inflation / dropout simulator:
      - For each gene j, estimate its mean expression m_j across cells.
      - Use an empirical logistic model to map log(m_j) to a dropout
        probability p_j, with lower expression -> higher dropout.
      - For a given global strength factor s, define
            p_j^(s) = clip(s * p_j, 0, 1)
      - Then for each cell i, gene j:
            Z_ij ~ Bernoulli(p_j^(s))
            X_ij = 0      if Z_ij = 1
            X_ij = y_ij   if Z_ij = 0
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Output directory ready: {output_dir} ---")

    print(f"--- Loading data: {file_path} ---")
    adata_original = anndata.read_h5ad(file_path)

    print("\n--- Original data layout ---")
    print(adata_original)

    is_sparse = sp.issparse(adata_original.X)
    print(f"\nexpression matrix (adata.X) is sparse: {is_sparse}")

    original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()
    original_data = np.maximum(original_data, 0.0)

    n_cells, n_genes = original_data.shape

    print("\n--- Computing per-gene mean expression (for dropout probability) ---")
    gene_means = np.asarray(original_data.mean(axis=0)).ravel()
    epsilon = 1e-8
    gene_means_safe = np.where(gene_means > 0, gene_means, epsilon)

    log_m = np.log1p(gene_means_safe)

    print("\n--- Building logistic dropout model from log(mean) ---")
    x1 = np.quantile(log_m, 0.1)
    x2 = np.quantile(log_m, 0.9)
    p1 = 0.8
    p2 = 0.2

    def logit_inv(p):
        return np.log(1.0 / p - 1.0)

    y1 = logit_inv(p1)
    y2 = logit_inv(p2)

    if x2 != x1:
        b = (y2 - y1) / (x2 - x1)
    else:
        b = 0.0
    a = y1 - b * x1

    print(f"Logistic params: a = {a:.6f}, b = {b:.6f}")
    print("Low-expression genes have higher dropout probability.")

    base_logits = a + b * log_m
    base_p = 1.0 / (1.0 + np.exp(base_logits))
    base_p = np.clip(base_p, 0.0, 1.0)

    strength_factors = [0.3, 0.6, 1.0, 1.6, 2.2]
    print(f"\n--- Dropout strength factors: {strength_factors} ---")

    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    for s in strength_factors:
        print(f"\n========== Processing zero-inflation level: strength = {s} ==========")

        p_level = np.clip(s * base_p, 0.0, 1.0)
        p_row = p_level.reshape(1, -1)

        print("--- Sampling Bernoulli dropout mask per cell-gene ---")
        dropout_mask = np.random.rand(n_cells, n_genes) < p_row

        noisy_data = original_data.copy()
        noisy_data[dropout_mask] = 0.0
        noisy_data = noisy_data.astype(np.float32)

        adata_noisy = anndata.AnnData(
            noisy_data,
            obs=adata_original.obs.copy(),
            var=adata_original.var.copy()
        )

        adata_noisy.uns["dropout_model"] = "ZeroInflation_Bernoulli"
        adata_noisy.uns["dropout_strength_factor"] = float(s)
        adata_noisy.uns["dropout_logistic_a"] = float(a)
        adata_noisy.uns["dropout_logistic_b"] = float(b)
        adata_noisy.uns["dropout_note"] = (
            "Counts set to zero via gene-wise Bernoulli dropout with "
            "p_j derived from logistic(log(mean_expression_j)), "
            "scaled by a global strength factor."
        )

        output_filename = f"{base_filename}_zeroinflation_strength_{s}.h5ad"
        output_path = os.path.join(output_dir, output_filename)
        print(f"Saving to: {output_path}")
        adata_noisy.write_h5ad(output_path)

    print("\n--- All zero-inflation / dropout datasets generated ---")


if __name__ == "__main__":
    input_file = "/share/PertBench/data/fig1/raw_task1/task1_train_CD4T_exp.h5ad"

    if not os.path.exists(input_file):
        print(f"Not found '{input_file}'. Creating dummy h5ad for testing...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.poisson(lam=5.0, size=(n_obs, n_vars)).astype(np.float32)
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f"Created dummy file '{input_file}'.")

    output_directory = "/share/PertBench/data/add_zero_inflation_output"

    generate_zeroinflated_dropout_files(input_file, output_directory)
