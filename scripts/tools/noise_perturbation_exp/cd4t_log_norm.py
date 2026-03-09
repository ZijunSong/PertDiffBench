#!/usr/bin/env python
# -*- coding: utf-8 -*-

import anndata
import numpy as np
import scipy.sparse as sp
import os


def generate_lognormal_bionoise_files(file_path, output_dir):
    """
    使用对数正态分布 + Poisson 采样来模拟 cell-to-cell 生物变异，基于原始 h5ad 数据构造多个不同 CV 水平的扰动数据。

    Biological noise simulator:
      1) For each gene, take its mean expression m_j across cells.
      2) For each target coefficient of variation (CV = c), assume:
            X_ij ~ LogNormal(mu_j(c), sigma(c)^2)
         where
            sigma(c)^2 = ln(c^2 + 1)
            mu_j(c)    = ln(m_j) - 0.5 * sigma(c)^2
      3) Then sample observed counts via:
            K_ij ~ Poisson(X_ij)

    这样就实现了“Log-normal 生成潜在表达率 + Poisson 计数采样”的 BigSur 样式生物噪声模型。
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

    # 3. Compute per-gene means across cells as baseline expression
    #    shape: (n_vars, )
    print("\n--- 正在计算每个基因的平均表达量 (作为 Log-normal 的目标均值) ---")
    gene_means = np.asarray(original_data.mean(axis=0)).ravel()

    # To avoid log(0), replace non-positive means with a small epsilon
    epsilon = 1e-3
    gene_means_safe = np.where(gene_means > 0, gene_means, epsilon)

    # 4. Define different CV (coefficient of variation) levels for biological noise
    #    You can tune this list according to your experiment design.
    cv_levels = [0.1, 0.25, 0.5, 1.0, 1.5]
    print(f"\n--- 将要使用的生物噪声 CV (coefficient of variation) 列表: {cv_levels} ---")

    n_cells, n_genes = original_data.shape
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # 5. Loop over CV levels and generate synthetic datasets
    for c in cv_levels:
        print(f"\n========== 正在处理生物噪声级别: CV = {c} ==========")

        # For LogNormal, with specified mean m and CV = c:
        #   CV^2 + 1 = exp(sigma^2)  =>  sigma^2 = ln(CV^2 + 1)
        #   mu = ln(m) - 0.5 * sigma^2
        sigma2 = np.log(c ** 2 + 1.0)
        sigma = np.sqrt(sigma2)
        mu = np.log(gene_means_safe) - 0.5 * sigma2

        print(f"当前 CV = {c} 对应的 sigma^2 = {sigma2:.6f}, sigma = {sigma:.6f}")
        print("mu 采用每个基因的 ln(mean) 做平移（gene-specific mu_j）")

        # Prepare broadcasting shapes:
        # mu: (1, n_genes)  -> broadcast to (n_cells, n_genes)
        mu_row = mu.reshape(1, -1)

        # 5.1 Sample latent expression rates from LogNormal
        #     X_ij ~ LogNormal(mu_j, sigma^2)
        print("--- 从 Log-normal 分布采样潜在表达率矩阵 X (可能会稍微慢一点) ---")
        latent_rates = np.random.lognormal(mean=mu_row, sigma=sigma, size=(n_cells, n_genes))

        # 5.2 Sample observed counts via Poisson
        #     K_ij ~ Poisson(X_ij)
        print("--- 从 Poisson 分布根据 X 采样观测计数矩阵 K ---")
        noisy_counts = np.random.poisson(lam=latent_rates).astype(np.float32)

        # 5.3 Construct new AnnData object
        adata_noisy = anndata.AnnData(
            noisy_counts,
            obs=adata_original.obs.copy(),
            var=adata_original.var.copy()
        )

        # You may want to record metadata about how this dataset was generated
        adata_noisy.uns["bionoise_model"] = "LogNormal+Poisson"
        adata_noisy.uns["bionoise_cv"] = float(c)
        adata_noisy.uns["bionoise_sigma2"] = float(sigma2)
        adata_noisy.uns["bionoise_note"] = (
            "Counts generated via gene-wise LogNormal(mu_j(c), sigma(c)^2) "
            "followed by Poisson sampling, using original gene means as targets."
        )

        # 5.4 Save to new h5ad file
        output_filename = f"{base_filename}_lognorm_cv_{c}.h5ad"
        output_path = os.path.join(output_dir, output_filename)
        print(f"正在保存到: {output_path}")
        adata_noisy.write_h5ad(output_path)

    print("\n--- 所有基于对数正态分布的生物噪声模拟数据已成功生成！ ---")


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

    output_directory = "/share/PertBench/data/add_lognormal_bionoise_output"

    generate_lognormal_bionoise_files(input_file, output_directory)
