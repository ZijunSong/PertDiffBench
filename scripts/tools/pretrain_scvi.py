# scripts/pretrain_scvi.py

import os
import scanpy as sc
import numpy as np # 用于数据检查
import torch # 用于设置矩阵乘法精度
from scipy import sparse
import scvi

def main(exp_h5ad, save_dir, max_epochs=100):
    # PyTorch 建议：为兼容的 GPU 设置矩阵乘法精度以提高潜在性能。
    # 'high' 更安全稳定，'medium' 可能更快。
    # 这不太可能是 NaN 的主要原因，但是一个好的实践。
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 1) 读入原始 AnnData 对象
    print(f"正在从 {exp_h5ad} 读取 AnnData...")
    adata_orig = sc.read_h5ad(exp_h5ad)
    print(f"原始 AnnData: {adata_orig.n_obs} 个细胞 x {adata_orig.n_vars} 个基因")

    # 为 scVI 预处理创建一个工作副本
    adata = adata_orig.copy()

    # 2) 确保 adata.X 包含 scVI 所需的原始计数
    # scVI 期望原始计数。如果 adata.X 是标准化的，并且原始计数在 adata.raw 中：
    if adata.raw is not None:
        print("发现 adata.raw。正在检查 adata.X 是否与 adata.raw.X 不同。")
        print("将使用 adata.raw.X 中的计数数据进行 scVI 分析。")
        adata.X = adata.raw.X.copy()
    else:
        print("未发现 adata.raw。假设 adata.X 包含原始计数数据。"
              "请确保这是正确的（非负整数计数），否则 scVI 可能会产生 NaN 或不佳结果。")

    print(f"进行检查前，adata.X 的类型为: {type(adata.X)}")
    if sparse.issparse(adata.X):  # 如果是稀疏矩阵
        print(f"进行检查前，adata.X.data 的类型为: {type(adata.X.data)}")

    # 验证 adata.X 是否包含非负值，并且最好是整数
    # 对于稀疏矩阵，检查 .data 属性
    current_X_accessor = adata.X.data if sparse.issparse(adata.X) else adata.X

    # 显式转换为 NumPy 数组，以处理 memoryview 或其他非 NumPy 数组类型
    try:
        data_for_check = np.asarray(current_X_accessor)
    except Exception as e:
        print(f"错误：无法将 adata.X 或 adata.X.data (类型: {type(current_X_accessor)}) 转换为 NumPy 数组进行检查: {e}")
        return

    if np.any(data_for_check < 0):
        print("错误：在 adata.X 中发现负值。scVI 要求非负计数。")
        return

    # 检查数据是否包含非整数值 (例如浮点数)
    # np.array_equal(data_for_check, np.round(data_for_check)) 会检查所有元素是否与其四舍五入后的值相等
    is_already_whole_numbers = np.array_equal(data_for_check, np.round(data_for_check))

    if not is_already_whole_numbers:
        print("警告：adata.X 包含非整数的浮点值。scVI 通常期望整数计数。"
              "将对值进行四舍五入并转换为整数。请检查您的数据源是否确实是计数数据。")
        if sparse.issparse(adata.X):  # 稀疏矩阵
            adata.X.data = np.round(data_for_check).astype(np.int32) # 使用常见的整数类型
        else: # 稠密矩阵
            adata.X = np.round(data_for_check).astype(np.int32)
    # 如果数据已经是整数（例如 1.0, 2.0），但存储为浮点类型，或者不是期望的整数类型 (如 np.int32)
    elif not np.issubdtype(data_for_check.dtype, np.integer) or data_for_check.dtype != np.int32 :
        print("信息：adata.X 中的值是整数（或浮点型的整数），但其数据类型不是期望的 np.int32。将进行转换。")
        if sparse.issparse(adata.X):  # 稀疏矩阵
            adata.X.data = data_for_check.astype(np.int32)
        else: # 稠密矩阵
            adata.X = data_for_check.astype(np.int32)
    else:
        print("信息：adata.X 中的数据已为期望的整数类型 (np.int32) 且非负。")

    # 打印修改后数据类型的日志
    if sparse.issparse(adata.X):
        print(f"检查和潜在转换后，adata.X.data 的数据类型: {adata.X.data.dtype}")
    else:
        print(f"检查和潜在转换后，adata.X 的数据类型: {adata.X.dtype}")


    # 3) 根据 scVI 警告，过滤掉没有检测到基因的细胞（空细胞）
    print(f"过滤空细胞前细胞数量: {adata.n_obs}")
    sc.pp.filter_cells(adata, min_genes=1) # 保留至少表达1个基因的细胞
    print(f"过滤空细胞后细胞数量: {adata.n_obs}")

    if adata.n_obs == 0:
        print("错误：在 'min_genes=1' 过滤后，所有细胞都被过滤掉了。"
              "您的数据集可能是空的或基因检测非常稀疏。请检查您的输入数据。")
        return

    # 4) 过滤掉在任何细胞中均未检测到的基因（可选，但推荐）
    print(f"过滤未表达基因前基因数量: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=1) # 保留至少在1个细胞中表达的基因
    print(f"过滤未表达基因后基因数量: {adata.n_vars}")

    if adata.n_vars == 0:
        print("错误：在 'min_cells=1' 过滤后，所有基因都被过滤掉了。"
              "您的数据集中可能没有基因在细胞中表达。请检查您的输入数据。")
        return

    # 5) 筛选高变基因
    print(f"筛选高变基因前基因数量: {adata.n_vars}")
    # adata.X：你可以按需做归一化、对数化等
    # adata.layers["counts"]：始终保留原始整数 counts
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, layer="counts", flavor="seurat_v3")
    print(f"筛选高变基因后基因数量: {adata.n_vars}")

    # 6) 为 scvi-tools 设置 AnnData
    # 此步骤注册计数数据。如果您有批次信息（例如，在 adata.obs['batch_column'] 中），
    # 您应该在此处指定：scvi.model.SCVI.setup_anndata(adata, batch_key='batch_column')
    print("正在为 scvi-tools 设置 AnnData...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Condition"])

    # 7) 初始化并训练 scVI 模型
    print("正在初始化 scVI 模型...")
    # n_latent=128 是一个常见的选择，但可以调整。
    vae = scvi.model.SCVI(adata, n_latent=128)

    print(f"开始模型训练，共 {max_epochs} 个周期...")

    try:
        # 如果您有验证集，可以考虑 check_val_every_n_epoch 进行监控。
        # 默认批次大小为 128。如果数据集非常小，这可能会有问题。
        # 之前的错误追踪显示总共有 4 个批次，这意味着 数据集大小 / 批次大小 = 4。
        vae.train(max_epochs=max_epochs, check_val_every_n_epoch=10) # 每 10 个周期记录一次验证损失
    except ValueError as e:
        print(f"训练过程中发生 ValueError: {e}")
        print("如果数据问题仍然存在（例如，在设置后批处理中全为零，或存在极端值），则可能发生这种情况。")
        print("正在将处理后的 AnnData 对象保存到 'debug_adata_before_scvi_train.h5ad' 以供检查。")
        adata.write_h5ad("debug_adata_before_scvi_train.h5ad")
        return
    except Exception as e: # 捕获任何其他异常
        print(f"训练过程中发生意外错误: {e}")
        print("正在将处理后的 AnnData 对象保存到 'debug_adata_before_scvi_train.h5ad' 以供检查。")
        adata.write_h5ad("debug_adata_before_scvi_train.h5ad")
        return

    # 8) 保存模型（这将在 `save_dir` 目录下创建 model.pt 等文件）
    print(f"训练完成。正在将模型保存到 {save_dir}...")
    adata.write_h5ad(os.path.join(save_dir, "adata_scvi_for_inference.h5ad"))
    vae.save(save_dir, overwrite=True) # overwrite=True 对于迭代运行很方便

    print(f"成功将 scVI 模型保存到 {save_dir}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="预训练 scVI 模型。")
    p.add_argument("--h5ad", default="dataset/scrna_data/scrna_positive.h5ad",
                   help="输入 AnnData 文件的路径 (应包含原始计数)。")
    p.add_argument("--out", default="checkpoints/scvi_model",
                   help="保存训练好的 scVI 模型的目录。")
    p.add_argument("--epochs", type=int, default=500,
                   help="训练周期数。")
    args = p.parse_args()

    main(args.h5ad, args.out, args.epochs)