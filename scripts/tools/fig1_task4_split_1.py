import sys
import os
import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

def get_label_from_filename(filepath):
    """
    根据文件名中的关键词智能分配指定的标签。
    - 如果文件名含 'coculture' 或 'ifn'，标签为 'IFN'。
    - 如果文件名含 'control'，标签为 'Control'。
    """
    # 获取小写的文件名，不含路径和扩展名，便于关键词匹配
    filename_lower = os.path.basename(filepath).lower().replace('.h5ad', '')

    if 'coculture' in filename_lower:
        return 'IFN'
    elif 'ifn' in filename_lower:
        return 'IFN'
    elif 'control' in filename_lower:
        return 'Control'
    else:
        # 如果没有找到指定关键词，打印一个警告并使用备用逻辑
        print(f"警告: 在文件名 '{os.path.basename(filepath)}' 中未找到 'control', 'coculture', 或 'ifn'。")
        print("将使用默认逻辑（文件名最后一个 '_' 后的部分）作为标签。")
        label = filename_lower.split('_')[-1].capitalize()
        return label

def merge_and_split_data(file1_path, file2_path, output_train_path, output_test_path):
    """
    合并两个h5ad文件，然后按80/20的比例分层抽样分割为训练集和测试集。
    """
    print("--- 开始执行合并与分割任务 ---")

    # --- 1. 读取 H5AD 文件 ---
    try:
        print(f"读取文件 1: {file1_path}")
        adata1 = ad.read_h5ad(file1_path)
        print(f"读取文件 2: {file2_path}")
        adata2 = ad.read_h5ad(file2_path)
    except Exception as e:
        print(f"错误: 读取文件时失败 - {e}")
        sys.exit(1)

    # --- 2. 自动生成标签并添加到 .obs ---
    label1 = get_label_from_filename(file1_path)
    label2 = get_label_from_filename(file2_path)
    print(f"\n--- 自动生成标签 ---")
    print(f"文件1 '{os.path.basename(file1_path)}' 的标签被指定为: '{label1}'")
    print(f"文件2 '{os.path.basename(file2_path)}' 的标签被指定为: '{label2}'")

    adata1.obs['perturbation_status'] = label1
    adata2.obs['perturbation_status'] = label2

    # --- 3. 合并两个 AnnData 对象 ---
    print("\n--- 正在合并数据 ---")
    # 添加 fill_value=0 参数
    merged_adata = ad.concat([adata1, adata2], join='outer', fill_value=0)
    print(f"合并完成，共计 {merged_adata.n_obs} 个细胞。")
    print("合并后各类别细胞数量:")
    print(merged_adata.obs['perturbation_status'].value_counts())

    merged_adata.obs.rename(columns={'celltype': 'Cell.Type'}, inplace=True)
    
    # --- 4. 按8:2比例进行分层抽样分割 ---
    print("\n--- 正在按 80/20 比例分割数据集 ---")

    indices = np.arange(merged_adata.n_obs)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=merged_adata.obs['perturbation_status']
    )

    adata_train = merged_adata[train_indices, :]
    adata_test = merged_adata[test_indices, :]

    # --- 5. 验证分割结果 ---
    print("\n--- 验证分割结果 ---")
    print(f"训练集形状: {adata_train.shape}")
    print("训练集中各类别细胞数量:")
    print(adata_train.obs['perturbation_status'].value_counts())

    print(f"\n测试集形状: {adata_test.shape}")
    print("测试集中各类别细胞数量:")
    print(adata_test.obs['perturbation_status'].value_counts())

    # --- 6. 保存训练集和测试集 ---
    try:
        print(f"\n--- 正在保存文件 ---")
        adata_train.write_h5ad(output_train_path)
        print(f"训练集已保存到: '{output_train_path}'")
        adata_test.write_h5ad(output_test_path)
        print(f"测试集已保存到: '{output_test_path}'")
        print("\n✅ 任务成功完成！")
    except Exception as e:
        print(f"错误: 保存文件时失败 - {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("\n[错误] 参数数量不正确！")
        print("用法: python merge_and_split.py <输入文件1> <输入文件2> <输出训练集路径> <输出测试集路径>")
        print("\n示例:")
        print("  python merge_and_split.py task4_ACTA2_control.h5ad task4_ACTA2_coculture.h5ad train_data.h5ad test_data.h5ad")
        sys.exit(1)

    file1, file2, out_train, out_test = sys.argv[1:5]
    merge_and_split_data(file1, file2, out_train, out_test)