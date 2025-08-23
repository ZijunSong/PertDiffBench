import pandas as pd
import anndata as ad
import numpy as np
import os
import glob
import scanpy as sc

# 定义H5AD文件所在的目录（假设您已运行上一个脚本生成了H5AD文件）
data_dir = 'data/fig1/task1/'
# 定义输出高变基因文件的目录
output_dir = 'data/fig1/task1_hvg_output/'
os.makedirs(output_dir, exist_ok=True) # 如果目录不存在则创建

print(f"开始处理目录中的H5AD文件: {data_dir}...")

# 获取所有H5AD文件
h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))

if not h5ad_files:
    print(f"错误：在 {data_dir} 中没有找到任何H5AD文件。请确保您已运行前面的脚本生成了H5AD文件。")
    exit()

# 定义目标细胞类型和数据集类型
target_cell_type = 'NK'
target_dataset_types = ['train', 'valid'] # 现在同时处理 train 和 test

# 定义高变基因数量的梯度
hvg_numbers = [1000, 2000, 3000, 4000, 5000, 6000]

# 遍历每个目标数据集类型 (train 和 valid)
for current_dataset_type in target_dataset_types:
    print(f"\n--- 正在处理 '{target_cell_type}' 细胞类型的 '{current_dataset_type}' 数据集 ---")

    selected_h5ad_file = None
    for h5ad_file in h5ad_files:
        base_name = os.path.basename(h5ad_file).lower() # 转换为小写进行匹配
        # 查找同时包含目标细胞类型和当前数据集类型的文件
        if target_cell_type.lower() in base_name and current_dataset_type.lower() in base_name:
            selected_h5ad_file = h5ad_file
            break # 找到第一个匹配的文件就退出

    if selected_h5ad_file is None:
        print(f"错误：未找到符合 '{target_cell_type}' 和 '{current_dataset_type}' 条件的H5AD文件。跳过此数据集。")
        continue # 跳过当前数据集，处理下一个

    print(f"已选择文件: {os.path.basename(selected_h5ad_file)}")

    try:
        adata_selected_cell_type = ad.read_h5ad(selected_h5ad_file)
        print("AnnData文件读取成功。")
        print(f"数据维度 (细胞 x 基因): {adata_selected_cell_type.shape}")
        # 验证细胞类型是否与预期一致 (可选，但推荐)
        if 'Cell.Type' in adata_selected_cell_type.obs and adata_selected_cell_type.obs['Cell.Type'].iloc[0] != target_cell_type:
            print(f"警告: 加载文件的实际细胞类型 '{adata_selected_cell_type.obs['Cell.Type'].iloc[0]}' 与目标 '{target_cell_type}' 不符。请检查文件。")

    except Exception as e:
        print(f"读取H5AD文件 {selected_h5ad_file} 时出错: {e}")
        continue # 跳过当前数据集，处理下一个

    # 为高变基因选择进行数据预处理：标准化和对数转换
    # 注意：这里我们对一个副本进行操作，以保留原始数据
    adata_hvg_processed = adata_selected_cell_type.copy()

    print("正在进行数据标准化和对数转换...")
    # 将每个细胞的总计数标准化到10,000
    sc.pp.normalize_total(adata_hvg_processed, target_sum=1e4)
    # 对数据进行对数转换
    sc.pp.log1p(adata_hvg_processed)
    print("标准化和对数转换完成。")

    print(f"--- 正在针对 '{current_dataset_type}' 数据集生成不同数量的高变基因文件 ---")

    # 计算高变基因。我们一次性计算出最多需要的高变基因（6000个），然后进行子集选择。
    # 如果总基因数少于6000，则会使用所有可用基因。
    num_genes_in_data = adata_hvg_processed.shape[1]
    n_top_genes_to_calculate = min(max(hvg_numbers), num_genes_in_data)

    print(f"正在计算高变基因，目标数量: {n_top_genes_to_calculate} (或所有基因，如果少于此数)...")
    # 使用 'seurat_v3' 方法计算高变基因，并将结果存储在 adata_hvg_processed.var 中
    sc.pp.highly_variable_genes(adata_hvg_processed, n_top_genes=n_top_genes_to_calculate, flavor='seurat_v3', subset=False)
    print("高变基因计算完成。")

    # 根据梯度的数量，生成并保存高变基因文件
    for n_hvg in hvg_numbers:
        if n_hvg > num_genes_in_data:
            print(f"警告: 请求的高变基因数 {n_hvg} 大于总基因数 {num_genes_in_data}。将使用所有基因。")
            current_n_hvg = num_genes_in_data
        else:
            current_n_hvg = n_hvg

        # 根据 'highly_variable_rank' 排序，选择前 'current_n_hvg' 个高变基因的索引
        top_hvg_genes_indices = adata_hvg_processed.var.sort_values('highly_variable_rank').index[:current_n_hvg]

        # 从原始（未标准化）的adata_selected_cell_type中选择这些高变基因
        # 这样做是为了确保输出文件包含原始计数数据，方便后续的灵活分析
        adata_filtered_hvg = adata_selected_cell_type[:, top_hvg_genes_indices].copy()

        # 构建输出文件名，包含数据集类型 (train/test)
        output_filename = os.path.join(output_dir, f"{target_cell_type}_{current_dataset_type}_HVG_{n_hvg}.h5ad")

        try:
            adata_filtered_hvg.write(output_filename)
            print(f"已成功保存文件: {output_filename} (包含 {adata_filtered_hvg.shape[1]} 个高变基因)")
        except Exception as e:
            print(f"保存文件 {output_filename} 时出错: {e}")

print("\n------ 所有高变基因文件生成完成 ------")
