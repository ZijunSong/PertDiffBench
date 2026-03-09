# create_h5ad.py

import pandas as pd
import anndata as ad
import argparse
import sys
import os

def verify_h5ad(file_path):
    """
    读取一个 .h5ad 文件并打印其关键信息以供验证。
    """
    if not os.path.exists(file_path):
        print(f"ERROR: 验证失败，文件 '{file_path}' 不存在。")
        return
        
    print("\n--- [验证步骤] ---")
    print(f"INFO: 正在重新读取已保存的文件 '{file_path}' 进行检查...")
    
    try:
        adata_check = ad.read_h5ad(file_path)
        
        print("\n1. AnnData 对象摘要:")
        print(adata_check)
        
        print("\n2. 细胞元数据 (obs) 的前5行:")
        print(adata_check.obs.head())
        
        print("\n3. 基因元数据 (var) 的前5行:")
        print(adata_check.var.head())

        print(f"\nINFO: 验证成功。数据集包含 {adata_check.n_obs} 个细胞和 {adata_check.n_vars} 个基因。")
        print("--- [验证结束] ---\n")

    except Exception as e:
        print(f"ERROR: 验证过程中读取或检查文件 '{file_path}' 时发生错误: {e}")


def create_h5ad(meta_path, exp_path, output_path):
    """
    将 meta.csv 和 exp.csv 文件合并成一个 h5ad 文件。

    Args:
        meta_path (str): meta.csv 文件的路径 (细胞元数据)。
        exp_path (str): exp.csv 文件的路径 (表达矩阵, 格式为 细胞 x 基因)。
        output_path (str): 输出的 .h5ad 文件的保存路径。
    """
    try:
        # --- 步骤 1: 加载细胞元数据 (obs) ---
        print(f"INFO: 正在从 '{meta_path}' 读取元数据...")
        # CSV的第一列是细胞ID，我们将其作为索引列
        meta_df = pd.read_csv(meta_path, index_col=0)

        # --- 步骤 2: 根据你的要求添加 'Cell.Type' 列 ---
        # 为所有细胞的 'Cell.Type' 列统一赋值为字符串 'species'
        meta_df['Cell.Type'] = 'species'
        print("INFO: 已成功创建 'Cell.Type' 列并统一赋值为 'species'。")

        # --- 步骤 3: 加载基因表达矩阵 (X) ---
        print(f"INFO: 正在从 '{exp_path}' 读取表达矩阵...")
        # CSV的第一列是细胞ID，作为索引；其余列是基因名
        exp_df = pd.read_csv(exp_path, index_col=0)

        # --- 步骤 4: 确保细胞ID（索引）对齐 ---
        # AnnData 要求 obs 和 X 的索引必须完全一致且顺序相同。
        # 我们使用两个文件索引的交集来确保数据对齐。
        common_cells = meta_df.index.intersection(exp_df.index)

        if len(common_cells) == 0:
            print("ERROR: 元数据文件和表达矩阵文件之间没有共同的细胞ID，无法继续。")
            sys.exit(1)
        
        # 如果两个文件的细胞不完全匹配，则发出警告
        if len(common_cells) < len(meta_df.index) or len(common_cells) < len(exp_df.index):
            print(f"WARNING: 并非所有细胞都同时存在于两个文件中。将使用 {len(common_cells)} 个共同细胞进行合并。")
        
        # 根据共同的细胞ID和顺序对齐两个DataFrame
        meta_df_aligned = meta_df.loc[common_cells]
        exp_df_aligned = exp_df.loc[common_cells]

        # --- 步骤 5: 创建 AnnData 对象 ---
        print("INFO: 正在创建 AnnData 对象...")
        # X: 表达矩阵 (细胞 x 基因)
        # obs: 细胞元数据
        # var: 基因元数据 (会自动从表达矩阵的列名创建)
        adata = ad.AnnData(
            X=exp_df_aligned,
            obs=meta_df_aligned
        )

        # --- 步骤 6: 保存为 h5ad 文件 ---
        print(f"INFO: 正在将 AnnData 对象保存到 '{output_path}'...")
        # 使用 gzip 压缩可以有效减小文件大小
        adata.write_h5ad(output_path, compression="gzip")

        print("\n🎉 处理完成！")
        
        # --- 步骤 7: 验证已保存的文件 ---
        verify_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: 文件未找到 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 处理过程中发生错误 - {e}")
        sys.exit(1)


if __name__ == '__main__':
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="将细胞元数据 (meta.csv) 和基因表达矩阵 (exp.csv) 合并为 h5ad 文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'meta_file', 
        help='输入的元数据CSV文件路径 (例如: meta.csv)。'
    )
    parser.add_argument(
        'exp_file', 
        help='输入的表达矩阵CSV文件路径 (例如: exp.csv)。'
    )
    parser.add_argument(
        'output_file', 
        help='输出的 .h5ad 文件路径 (例如: output.h5ad)。'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数执行转换
    create_h5ad(args.meta_file, args.exp_file, args.output_file)
