import anndata as ad
import numpy as np

# --- 加载所有最原始的 h5ad 文件 ---
control_file = 'data/fig1/task4/task4_ACTA2_control.h5ad'
coculture_file = 'data/fig1/task4/task4_ACTA2_coculture.h5ad'
ifn_file = 'data/fig1/task4/task4_ACTA2_ifn.h5ad'

adata_ctrl = ad.read_h5ad(control_file)
adata_cocul = ad.read_h5ad(coculture_file)
adata_ifn = ad.read_h5ad(ifn_file)

# --- 获取并合并所有基因列表 ---
genes_ctrl = set(adata_ctrl.var_names)
genes_cocul = set(adata_cocul.var_names)
genes_ifn = set(adata_ifn.var_names)

# 计算并集
global_gene_set = genes_ctrl.union(genes_cocul).union(genes_ifn)

# 转换为排序后的列表，以保证顺序一致性
global_gene_list = sorted(list(global_gene_set))

# --- 保存这个全局列表以备后用 ---
global_list_path = 'data/fig1/task4/global_gene_list.txt'
with open(global_list_path, 'w') as f:
    for gene in global_gene_list:
        f.write(f"{gene}\n")

print(f"✅ 全局基因列表已创建并保存。")
print(f"总基因数量: {len(global_gene_list)}")