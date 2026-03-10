import sys
import os
import anndata as ad
import pandas as pd

# --- 先加载全局基因列表 ---
try:
    with open('data/fig1/task4/global_gene_list.txt', 'r') as f:
        GLOBAL_GENE_LIST = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("❌ 错误: 'global_gene_list.txt' 未找到。请先运行步骤1的脚本。")
    sys.exit(1)


def merge_pair(file1_path, label1, file2_path, label2, output_path):
    # ... (函数内部的其他逻辑不变) ...
    print("-" * 50)
    print(f"处理任务: {os.path.basename(file1_path)} + {os.path.basename(file2_path)} -> {os.path.basename(output_path)}")
    
    adata1 = ad.read_h5ad(file1_path)
    adata2 = ad.read_h5ad(file2_path)
    adata1.obs['perturbation_status'] = label1
    adata2.obs['perturbation_status'] = label2
    merged_adata = ad.concat([adata1, adata2], join='outer', fill_value=0)

    # ✅ 核心改动：使用全局基因列表对齐数据
    print(f"  - 合并后形状: {merged_adata.shape}")
    print(f"  - 正在对齐到 {len(GLOBAL_GENE_LIST)} 个全局基因...")
    merged_adata = ad.AnnData(merged_adata.to_df().reindex(columns=GLOBAL_GENE_LIST, fill_value=0), obs=merged_adata.obs)
    print(f"  - 对齐后形状: {merged_adata.shape}")

    if 'celltype' in merged_adata.obs.columns:
        merged_adata.obs.rename(columns={'celltype': 'Cell.Type'}, inplace=True)
        print("  - 'celltype' 列已重命名为 'Cell.Type'。")

    merged_adata.write_h5ad(output_path)
    print(f"  ✅ 成功！文件已保存到: {output_path}")
    print("-" * 50)


if __name__ == '__main__':
    # ... (主函数逻辑不变) ...
    if len(sys.argv) != 6:
        print("\n[错误] 参数数量不正确！")
        print("\n用法: python create_merged_datasets.py <control文件> <coculture文件> <ifn文件> <输出control-coculture文件> <输出control-ifn文件>")
        sys.exit(1)

    control_file, coculture_file, ifn_file, output_coculture_path, output_ifn_path = sys.argv[1:6]
    
    merge_pair(control_file, 'Control', coculture_file, 'IFN', output_coculture_path)
    merge_pair(control_file, 'Control', ifn_file, 'IFN', output_ifn_path)