# merge_h5ad.py

import anndata as ad
import argparse
import sys
import os

def verify_merged_h5ad(file_path):
    """
    读取一个合并后的 .h5ad 文件并打印其关键信息以供验证。
    """
    if not os.path.exists(file_path):
        print(f"ERROR: 验证失败，文件 '{file_path}' 不存在。")
        return
        
    print("\n--- [验证步骤] ---")
    print(f"INFO: 正在重新读取已保存的文件 '{file_path}' 进行检查...")
    
    try:
        adata_check = ad.read_h5ad(file_path)
        
        print("\n1. 合并后 AnnData 对象摘要:")
        print(adata_check)
        
        print("\n2. 检查 'perturbation_status' 列:")
        if 'perturbation_status' in adata_check.obs.columns:
            print("INFO: 'perturbation_status' 列存在。")
            # 统计并打印该列中各类别的数量
            status_counts = adata_check.obs['perturbation_status'].value_counts()
            print("各状态下的细胞数量:")
            print(status_counts)
        else:
            print("WARNING: 未找到 'perturbation_status' 列！")

        print("\n3. 细胞元数据 (obs) 的前5行:")
        print(adata_check.obs.head())
        
        print("\n4. 细胞元数据 (obs) 的后5行 (以检查另一部分数据):")
        print(adata_check.obs.tail())

        print(f"\nINFO: 验证成功。最终数据集包含 {adata_check.n_obs} 个细胞和 {adata_check.n_vars} 个基因。")
        print("--- [验证结束] ---\n")

    except Exception as e:
        print(f"ERROR: 验证过程中读取或检查文件 '{file_path}' 时发生错误: {e}")


def merge_h5ad_files(control_path, perturbed_path, output_path):
    """
    合并两个h5ad文件，并添加一个obs列来区分它们。

    Args:
        control_path (str): 控制组 .h5ad 文件的路径。
        perturbed_path (str): 扰动组 .h5ad 文件的路径。
        output_path (str): 输出的合并后的 .h5ad 文件的保存路径。
    """
    try:
        # --- 步骤 1: 加载两个h5ad文件 ---
        print(f"INFO: 正在从 '{control_path}' 加载控制组数据...")
        adata_control = ad.read_h5ad(control_path)

        print(f"INFO: 正在从 '{perturbed_path}' 加载扰动组数据...")
        adata_perturbed = ad.read_h5ad(perturbed_path)

        # --- 步骤 2: 添加 'perturbation_status' 列 ---
        adata_control.obs['perturbation_status'] = 'Control'
        print("INFO: 已为控制组数据添加 'perturbation_status' = 'Control'。")

        adata_perturbed.obs['perturbation_status'] = 'IFN'
        print("INFO: 已为扰动组数据添加 'perturbation_status' = 'IFN'。")
        
        # --- 步骤 3: 合并两个AnnData对象 ---
        # AnnData.concatenate 默认使用 'inner' join，这意味着它只会保留
        # 在两个数据对象中都存在的基因（变量），这通常是期望的行为。
        print("INFO: 正在合并两个 AnnData 对象...")
        adata_merged = adata_control.concatenate(
            adata_perturbed,
            join='inner' # 可以是 'inner' 或 'outer'
        )

        print(f"INFO: 合并完成。新的数据集包含 {adata_merged.n_obs} 个细胞和 {adata_merged.n_vars} 个基因。")

        # --- 步骤 4: 保存合并后的文件 ---
        print(f"INFO: 正在将合并后的数据保存到 '{output_path}'...")
        adata_merged.write_h5ad(output_path, compression="gzip")
        
        print("\n🎉 处理完成！")

        # --- 步骤 5: 验证已保存的文件 ---
        verify_merged_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: 文件未找到 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 处理过程中发生错误 - {e}")
        sys.exit(1)


if __name__ == '__main__':
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="合并两个 .h5ad 文件，并添加 'perturbation_status' 标签以区分来源。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'control_h5ad', 
        help='输入的控制组 .h5ad 文件路径。'
    )
    parser.add_argument(
        'perturbed_h5ad', 
        help='输入的扰动组 .h5ad 文件路径。'
    )
    parser.add_argument(
        'output_h5ad', 
        help='输出的合并后的 .h5ad 文件路径。'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数执行合并
    merge_h5ad_files(args.control_h5ad, args.perturbed_h5ad, args.output_h5ad)
