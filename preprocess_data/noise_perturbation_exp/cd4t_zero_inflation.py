#!/usr/bin/env python
# -*- coding: utf-8 -*-

import anndata
import numpy as np
import scipy.sparse as sp
import os


def generate_zeroinflated_dropout_files(file_path, output_dir):
    """
    使用基因特异的零膨胀 / dropout 模型来模拟技术性失捕（dropout）。

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

    This matches the idea:
        X_ij = 0       with probability p(gene_j)
        X_ij = y_ij    with probability 1 - p(gene_j)
    and p(gene_j) is determined from gene mean via a logistic-type model
    similar to Splatter-style dropout.
    """

    # 1. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- 输出文件将保存在: {output_dir} ---")

    # 2. Load original h5ad file
    print(f"--- 正在加载数据: {file_path} ---")
    adata_original = anndata.read_h5ad(file_path)

    print("\n--- 原始数据格式如下 ---")
    print(adata_original)

    is_sparse = sp.issparse(adata_original.X)
    print(f"\n表达矩阵 (adata.X) 是否为稀疏矩阵: {is_sparse}")

    # Convert to dense array if necessary
    # NOTE: For very large data, this may be memory-heavy.
    original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()

    # Enforce non-negativity (just in case)
    original_data = np.maximum(original_data, 0.0)

    n_cells, n_genes = original_data.shape

    # 3. Gene-wise mean expression as basis for dropout rate
    print("\n--- 正在计算每个基因的平均表达量 (用于估计 dropout 概率) ---")
    gene_means = np.asarray(original_data.mean(axis=0)).ravel()
    epsilon = 1e-8
    gene_means_safe = np.where(gene_means > 0, gene_means, epsilon)

    # Use log1p(mean) as input to logistic model
    log_m = np.log1p(gene_means_safe)

    # 4. Fit a simple logistic mapping from log(mean) to dropout prob
    #    We choose two anchor points:
    #       - low expression (10% quantile)  -> high dropout (p ≈ 0.8)
    #       - high expression (90% quantile) -> low dropout (p ≈ 0.2)
    print("\n--- 正在构建基于 log(mean) 的 Logistic dropout 模型 ---")
    x1 = np.quantile(log_m, 0.1)
    x2 = np.quantile(log_m, 0.9)
    p1 = 0.8
    p2 = 0.2

    # Logistic form: p(x) = 1 / (1 + exp(a + b x))
    # => ln(1/p - 1) = a + b x
    def logit_inv(p):
        return np.log(1.0 / p - 1.0)

    y1 = logit_inv(p1)
    y2 = logit_inv(p2)

    # Solve for b and a
    if x2 != x1:
        b = (y2 - y1) / (x2 - x1)
    else:
        # Pathological case: all log_m equal, fall back to constant dropout
        b = 0.0
    a = y1 - b * x1

    print(f"Logistic 参数: a = {a:.6f}, b = {b:.6f}")
    print("低表达基因 dropout 概率高，高表达基因 dropout 概率低。")

    # Base gene-specific dropout probability p_j
    base_logits = a + b * log_m
    base_p = 1.0 / (1.0 + np.exp(base_logits))

    # Clamp to [0, 1] for safety
    base_p = np.clip(base_p, 0.0, 1.0)

    # 5. Define global strength factors to control overall dropout level
    #    s < 1: milder dropout; s > 1: stronger dropout.
    strength_factors = [0.3, 0.6, 1.0, 1.6, 2.2]
    print(f"\n--- 将要使用的 dropout 强度倍数 (strength_factors): {strength_factors} ---")

    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # 6. Loop over strength factors and generate zero-inflated datasets
    for s in strength_factors:
        print(f"\n========== 正在处理零膨胀噪声级别: strength = {s} ==========")

        # Gene-specific dropout probability under this strength
        p_level = np.clip(s * base_p, 0.0, 1.0)

        # Reshape for broadcasting across cells
        p_row = p_level.reshape(1, -1)

        print("--- 正在为每个 cell-gene 对采样 Bernoulli dropout 掩码 ---")
        # dropout_mask[i, j] = True means we drop this count (set to 0)
        dropout_mask = np.random.rand(n_cells, n_genes) < p_row

        # Apply dropout
        noisy_data = original_data.copy()
        noisy_data[dropout_mask] = 0.0
        noisy_data = noisy_data.astype(np.float32)

        # Construct new AnnData object
        adata_noisy = anndata.AnnData(
            noisy_data,
            obs=adata_original.obs.copy(),
            var=adata_original.var.copy()
        )

        # Record metadata
        adata_noisy.uns["dropout_model"] = "ZeroInflation_Bernoulli"
        adata_noisy.uns["dropout_strength_factor"] = float(s)
        adata_noisy.uns["dropout_logistic_a"] = float(a)
        adata_noisy.uns["dropout_logistic_b"] = float(b)
        adata_noisy.uns["dropout_note"] = (
            "Counts set to zero via gene-wise Bernoulli dropout with "
            "p_j derived from logistic(log(mean_expression_j)), "
            "scaled by a global strength factor."
        )

        # Save to new h5ad file
        output_filename = f"{base_filename}_zeroinflation_strength_{s}.h5ad"
        output_path = os.path.join(output_dir, output_filename)
        print(f"正在保存到: {output_path}")
        adata_noisy.write_h5ad(output_path)

    print("\n--- 所有基于零膨胀 / dropout 的技术噪声模拟数据已成功生成！ ---")


if __name__ == "__main__":
    # Example usage: you can adapt these paths to your environment
    input_file = "/share/PertBench/data/fig1/raw_task1/task1_train_CD4T_exp.h5ad"

    # When the original file does not exist, create a dummy dataset for testing
    if not os.path.exists(input_file):
        print(f"未找到 '{input_file}'。正在创建一个用于测试的虚拟 h5ad 文件...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.poisson(lam=5.0, size=(n_obs, n_vars)).astype(np.float32)
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f"已创建虚拟文件 '{input_file}'。")

    output_directory = "/share/PertBench/data/add_zero_inflation_output"

    generate_zeroinflated_dropout_files(input_file, output_directory)
