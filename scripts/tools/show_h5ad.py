import anndata
import os
import pandas as pd
import numpy as np

def surgical_inspect_h5ad(file_path: str):
    """
    读取并详细展示 h5ad 文件的信息。
    通过直接修改内部 _obsp 存储来解决数据不一致的校验错误。

    Args:
        file_path (str): h5ad 文件的路径。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件不存在于 '{file_path}'")
        return

    try:
        # 读取 h5ad 文件到 AnnData 对象。此时校验尚未触发。
        adata = anndata.read_h5ad(file_path)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # --- 最终修复方案：直接对内部存储进行操作 ---
    # 我们访问底层的 _obsp 字典，以绕过会触发错误的公开 .obsp 属性
    if hasattr(adata, '_obsp') and adata._obsp is not None:
        n_obs = adata.n_obs
        # 创建一个副本进行迭代，因为不能在迭代过程中修改字典
        obsp_keys = list(adata._obsp.keys())
        
        for key in obsp_keys:
            value = adata._obsp[key]
            # 检查形状是否与 (n_obs, n_obs) 匹配
            if value.shape[0] != n_obs or value.shape[1] != n_obs:
                print(f"[内部清理] ._obsp['{key}'] 的形状为 {value.shape}, 与观测值数量 ({n_obs}) 不匹配。正在删除...")
                # 直接从内部字典中删除有问题的项
                del adata._obsp[key]

    print("-" * 50)
    print(f"文件路径: {file_path}")
    print("-" * 50)

    # --- 总体信息 ---
    print("\n[ 总体信息 ]")
    print("AnnData 对象摘要信息：")
    # 现在这个 print 语句可以安全执行了
    print(adata)

    # --- 数据矩阵 (X) ---
    print("\n" + "=" * 50)
    print("[ 数据矩阵 (X) ]")
    print(f"  形状: {adata.X.shape}")
    print(f"  数据类型: {adata.X.dtype}")
    try:
        x_preview = adata.X[:5, :5]
        if hasattr(x_preview, "toarray"):
            x_preview = x_preview.toarray()
        print("  前 5x5 数据预览:")
        with pd.option_context('display.max_rows', 5, 'display.max_columns', 5, 'display.width', 100):
            print(pd.DataFrame(x_preview))
    except Exception as e:
        print(f"  无法生成 X 的预览: {e}")

    # ... (后续代码与之前版本相同，这里为了简洁省略) ...

    # --- 观测值元数据 (obs) ---
    print("\n" + "=" * 50)
    print("[ 观测值元数据 (obs) - 关于细胞/样本的信息 ]")
    print(f"  形状: {adata.obs.shape}")
    print(f"  列名: {list(adata.obs.columns)}")
    print("  前 5 行预览:")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 1000):
        print(adata.obs.head())

    # --- 变量元数据 (var) ---
    print("\n" + "=" * 50)
    print("[ 变量元数据 (var) - 关于基因/特征的信息 ]")
    print(f"  形状: {adata.var.shape}")
    print(f"  列名: {list(adata.var.columns)}")
    print("  前 5 行预览:")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 1000):
        print(adata.var.head())

    # --- 多维观测值注释 (obsm) ---
    print("\n" + "=" * 50)
    print("[ 多维观测值注释 (obsm) - 降维结果等 ]")
    if adata.obsm: # 这里的 .obsm 访问现在是安全的
        for key, value in adata.obsm.items():
            try:
                print(f"  - '{key}': 形状 {value.shape}, 类型 {type(value)}")
            except AttributeError:
                 print(f"  - '{key}': 类型 {type(value)}")
    else:
        print("  (空)")

    # --- 非结构化注释 (uns) ---
    print("\n" + "=" * 50)
    print("[ 非结构化注释 (uns) - 其他信息 ]")
    if adata.uns:
        for key, value in adata.uns.items():
            value_repr = repr(value)
            if len(value_repr) > 70:
                value_repr = value_repr[:70] + "..."
            print(f"  - '{key}': 类型 {type(value)}, 值 (预览): {value_repr}")
    else:
        print("  (空)")

    # --- 层 (layers) ---
    print("\n" + "=" * 50)
    print("[ 层 (layers) - 其他数据矩阵 ]")
    if adata.layers:
        for key, value in adata.layers.items():
            print(f"  - '{key}': 形状 {value.shape}, 类型 {type(value)}")
    else:
        print("  (空)")

    print("\n" + "-" * 50)
    print("信息展示完毕。")
    print("-" * 50)


if __name__ == "__main__":
    h5ad_file_path = "/share/PertBench/data/add_gaussian_noise_output/task1_train_CD4T_exp_noise_std_0.5.h5ad"
    surgical_inspect_h5ad(h5ad_file_path)