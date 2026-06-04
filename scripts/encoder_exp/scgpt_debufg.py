from scgpt.tasks.cell_emb import embed_data
import scanpy as sc

adata = sc.read_h5ad("data/fig1/raw_task1/task1_train_CD4T_exp.h5ad")

# beforehandle
# - detect_gene_col
# - adata.obs["cell_type"] = "unknown"
# afterdirectly call embed_data, return 

result = embed_data(adata_or_file=adata, model_dir="...", device="cuda")
print(type(result))

# AnnData:
new_adata = result
print("X shape:", new_adata.X.shape)
print("obsm keys:", new_adata.obsm.keys())
print("layers keys:", new_adata.layers.keys())
