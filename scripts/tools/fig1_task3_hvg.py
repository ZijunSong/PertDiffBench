import pandas as pd
import anndata as ad
import numpy as np
import os
import glob
import scanpy as sc

data_dir = 'data/fig1/task3/'
output_dir = 'data/fig1/task3_hvg_output/'
os.makedirs(output_dir, exist_ok=True)

print(f"开始处理目录中的H5AD文件: {data_dir}...")

h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))

if not h5ad_files:
    print(f"错误：在 {data_dir} 中没有找到任何H5AD文件。请确保您已运行前面的脚本生成了H5AD文件。")
    exit()

target_cell_types = [f'mix{i}' for i in range(2, 8)]
target_dataset_types = ['train', 'test']
hvg_numbers = [1000, 2000, 3000, 4000, 5000, 6000]

for target_cell_type in target_cell_types:
    print(f"\n==================== 开始处理细胞类型: {target_cell_type} ====================")
    for current_dataset_type in target_dataset_types:
        print(f"\n--- 正在处理 '{target_cell_type}' 细胞类型的 '{current_dataset_type}' 数据集 ---")

        selected_h5ad_file = None
        for h5ad_file in h5ad_files:
            base_name = os.path.basename(h5ad_file).lower()
            if target_cell_type.lower() in base_name and current_dataset_type.lower() in base_name:
                selected_h5ad_file = h5ad_file
                break

        if selected_h5ad_file is None:
            print(f"错误：未找到符合 '{target_cell_type}' 和 '{current_dataset_type}' 条件的H5AD文件。跳过此数据集。")
            continue

        print(f"已选择文件: {os.path.basename(selected_h5ad_file)}")

        try:
            adata_selected_cell_type = ad.read_h5ad(selected_h5ad_file)
            print("AnnData文件读取成功。")
            print(f"数据维度 (细胞 x 基因): {adata_selected_cell_type.shape}")
            if 'Cell.Type' in adata_selected_cell_type.obs and adata_selected_cell_type.obs['Cell.Type'].iloc[0] != target_cell_type:
                print(f"警告: 加载文件的实际细胞类型 '{adata_selected_cell_type.obs['Cell.Type'].iloc[0]}' 与目标 '{target_cell_type}' 不符。请检查文件。")

        except Exception as e:
            print(f"读取H5AD文件 {selected_h5ad_file} 时出错: {e}")
            continue

        adata_hvg_processed = adata_selected_cell_type.copy()

        print("正在进行数据标准化和对数转换...")
        sc.pp.normalize_total(adata_hvg_processed, target_sum=1e4)
        sc.pp.log1p(adata_hvg_processed)
        print("标准化和对数转换完成。")

        print(f"--- 正在针对 '{current_dataset_type}' 数据集生成不同数量的高变基因文件 ---")

        num_genes_in_data = adata_hvg_processed.shape[1]
        n_top_genes_to_calculate = min(max(hvg_numbers), num_genes_in_data)

        print(f"正在计算高变基因，目标数量: {n_top_genes_to_calculate} (或所有基因，如果少于此数)...")
        sc.pp.highly_variable_genes(adata_hvg_processed, n_top_genes=n_top_genes_to_calculate, flavor='seurat_v3', subset=False)
        print("高变基因计算完成。")

        for n_hvg in hvg_numbers:
            if n_hvg > num_genes_in_data:
                print(f"警告: 请求的高变基因数 {n_hvg} 大于总基因数 {num_genes_in_data}。将使用所有基因。")
                current_n_hvg = num_genes_in_data
            else:
                current_n_hvg = n_hvg

            top_hvg_genes_indices = adata_hvg_processed.var.sort_values('highly_variable_rank').index[:current_n_hvg]
            
            adata_filtered_hvg = adata_selected_cell_type[:, top_hvg_genes_indices].copy()
            
            output_filename = os.path.join(output_dir, f"{target_cell_type}_{current_dataset_type}_HVG_{n_hvg}.h5ad")

            try:
                adata_filtered_hvg.write(output_filename)
                print(f"已成功保存文件: {output_filename} (包含 {adata_filtered_hvg.shape[1]} 个高变基因)")
            except Exception as e:
                print(f"保存文件 {output_filename} 时出错: {e}")

print("\n------ 所有高变基因文件生成完成 ------")
