import anndata as ad
import numpy as np

# --- all original h5ad file ---
control_file = 'data/fig1/task4/task4_ACTA2_control.h5ad'
coculture_file = 'data/fig1/task4/task4_ACTA2_coculture.h5ad'
ifn_file = 'data/fig1/task4/task4_ACTA2_ifn.h5ad'

adata_ctrl = ad.read_h5ad(control_file)
adata_cocul = ad.read_h5ad(coculture_file)
adata_ifn = ad.read_h5ad(ifn_file)

# --- getand andallgenecols ---
genes_ctrl = set(adata_ctrl.var_names)
genes_cocul = set(adata_cocul.var_names)
genes_ifn = set(adata_ifn.var_names)

# and 
global_gene_set = genes_ctrl.union(genes_cocul).union(genes_ifn)

# convert to aftercols , to 
global_gene_list = sorted(list(global_gene_set))

# --- cols to afterusing ---
global_list_path = 'data/fig1/task4/global_gene_list.txt'
with open(global_list_path, 'w') as f:
    for gene in global_gene_list:
        f.write(f"{gene}\n")

print(f"✅ global gene list and .")
print(f" genecountamount: {len(global_gene_list)}")