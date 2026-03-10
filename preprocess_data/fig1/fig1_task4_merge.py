# merge_to_h5ad.py

import sys
import pandas as pd
import anndata as ad

def merge_csv_to_h5ad(meta_file, exp_file, output_file):
    """
    将细胞元数据 (meta) 和基因表达矩阵 (exp) 的 CSV 文件合并成一个 H5AD 文件。

    参数:
    meta_file (str): 细胞元数据 CSV 文件的路径 (行: 细胞, 列: 元数据特征)。
    exp_file (str): 基因表达矩阵 CSV 文件的路径 (行: 基因, 列: 细胞)。
    output_file (str): 输出的 H5AD 文件的路径。
    """
    # --- 1. 读取数据 ---
    print(f"--- 正在读取文件 ---")
    try:
        # 假设两个文件的第一列都是索引列
        print(f"读取元数据文件: {meta_file}")
        meta_df = pd.read_csv(meta_file, index_col=0)
        
        print(f"读取表达矩阵文件: {exp_file}")
        exp_df = pd.read_csv(exp_file, index_col=0)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        sys.exit(1) # 退出脚本并返回错误码
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        sys.exit(1)

    print("\n--- 数据预览 ---")
    print(f"元数据 (meta) 形状: {meta_df.shape}")
    print(f"表达矩阵 (exp) 形状: {exp_df.shape}")

    # --- 2. 确保数据对齐 (重要步骤) ---
    print("\n--- 正在对齐细胞ID ---")
    # 找到元数据和表达矩阵中共有的细胞ID
    common_cells = meta_df.index.intersection(exp_df.columns)

    if len(common_cells) == 0:
        print("错误：表达矩阵的细胞ID (列名) 和元数据的细胞ID (索引) 没有任何交集。")
        print("请检查你的文件内容是否匹配。")
        sys.exit(1)
    
    if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.columns):
        print("警告: 并非所有细胞都能在两个文件中匹配上。仅使用共有的细胞。")
        print(f"共有 {len(common_cells)} 个细胞将被包含在最终文件中。")

    # 根据共有的细胞ID过滤并重新排序，确保顺序完全一致
    meta_df_aligned = meta_df.loc[common_cells]
    exp_df_aligned = exp_df[common_cells]

    # --- 3. 创建 AnnData 对象 ---
    # AnnData 需要一个 细胞 x 基因 (obs x var) 的表达矩阵。
    # 我们的 exp_df_aligned 是 基因 x 细胞，所以需要转置 (.T)。
    print("\n--- 正在创建 AnnData 对象 ---")
    adata = ad.AnnData(X=exp_df_aligned.T,  # 转置表达矩阵
                       obs=meta_df_aligned) # obs (observations) 是细胞元数据
    
    # anndata 会自动从转置后的表达矩阵的列中推断出 var (variables, 基因元数据)。

    # --- 4. 验证创建的对象 ---
    print("\n--- AnnData 对象信息 ---")
    print(adata)
    print(f"观测值 (细胞) 元数据 (obs) 列: {list(adata.obs.columns)}")
    print(f"变量 (基因) 索引 (var) 名称示例: {list(adata.var.index[:5])}")

    # --- 5. 保存为 H5AD 文件 ---
    print(f"\n--- 正在保存到 H5AD 文件 ---")
    try:
        adata.write_h5ad(output_file)
        print(f"\n✅ 成功！数据已合并并保存到 '{output_file}' 文件中。")
    except Exception as e:
        print(f"保存 H5AD 文件时发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # 检查命令行参数数量是否正确
    if len(sys.argv) != 4:
        print("用法: python merge_to_h5ad.py <meta_csv_path> <exp_csv_path> <output_h5ad_path>")
        print("示例: python merge_to_h5ad.py meta.csv exp.csv merged_data.h5ad")
        sys.exit(1)

    # 从命令行获取文件名
    meta_csv_path = sys.argv[1]
    exp_csv_path = sys.argv[2]
    output_h5ad_path = sys.argv[3]

    # 调用主函数执行合并操作
    merge_csv_to_h5ad(meta_csv_path, exp_csv_path, output_h5ad_path)