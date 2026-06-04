import os
import anndata as ad

# Same as fig1_task4_merge.sh: merged H5AD lives under data/fig1_task4
DATA_DIR = "/data/ppnm/data/PertDiffBench/data/fig1_task4"

# --- Load all source h5ad files ---
control_file = os.path.join(DATA_DIR, "task4_ACTA2_control.h5ad")
coculture_file = os.path.join(DATA_DIR, "task4_ACTA2_coculture.h5ad")
ifn_file = os.path.join(DATA_DIR, "task4_ACTA2_ifn.h5ad")

adata_ctrl = ad.read_h5ad(control_file)
adata_cocul = ad.read_h5ad(coculture_file)
adata_ifn = ad.read_h5ad(ifn_file)

# --- Collect and union all gene lists ---
genes_ctrl = set(adata_ctrl.var_names)
genes_cocul = set(adata_cocul.var_names)
genes_ifn = set(adata_ifn.var_names)

# Union of gene sets
global_gene_set = genes_ctrl.union(genes_cocul).union(genes_ifn)

# Sorted list for consistent ordering
global_gene_list = sorted(list(global_gene_set))

# --- Save global gene list for later alignment ---
global_list_path = os.path.join(DATA_DIR, "global_gene_list.txt")
with open(global_list_path, "w") as f:
    for gene in global_gene_list:
        f.write(f"{gene}\n")

print(f"Global gene list created and saved: {global_list_path}")
print(f"Total genes: {len(global_gene_list)}")
