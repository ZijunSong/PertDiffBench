import os
import sys
import pandas as pd
import argparse
from scipy import sparse
import scanpy as sc

parser = argparse.ArgumentParser(description="Preprocess h5ad for scFoundation encoder")

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

parser.add_argument('--system_path', type=str, required=True,
                    help='Path to scFoundation-main/preprocessing/')
parser.add_argument('--file_name', type=str, required=True,
                    help='File name inside system_path/data/')
parser.add_argument('--sparse_matrix', type=strict_str2bool, default=True,
                    help='Whether to convert sparse matrix to dense.')

args = parser.parse_args()

system_path = args.system_path.rstrip("/")
file_name = args.file_name
sparse_matrix = args.sparse_matrix

# Import scRNA_workflow from this preprocessing tree (works after server migration).
# Some installs use system_path/code; this repo keeps scRNA_workflow.py next to this script.
for _p in (system_path, os.path.join(system_path, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(system_path)

from scRNA_workflow import *

# paths — read input from system_path/data/<file_name> (caller copies h5ad there); write preprocessed h5ad under system_path/
data_path = os.path.join(system_path, "data", file_name)
output_dir = system_path
os.makedirs(os.path.join(system_path, "data"), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"[SCF-preprocess] Reading: {data_path}")

# load h5ad
adata = sc.read_h5ad(data_path)

# convert sparse matrix if needed
if sparse_matrix:
    X_df = pd.DataFrame(
        adata.X.toarray(),
        index=adata.obs.index.tolist(),
        columns=adata.var.index.tolist()
    )
else:
    X_df = pd.DataFrame(
        adata.X,
        index=adata.obs.index.tolist(),
        columns=adata.var.index.tolist()
    )

# load gene index file
gene_index_file = os.path.join(system_path, "OS_scRNA_gene_index.19264.tsv")
assert os.path.exists(gene_index_file), \
    f"[ERROR] Missing gene index file: {gene_index_file}"

gene_list_df = pd.read_csv(gene_index_file, sep="\t")
gene_list = list(gene_list_df["gene_name"])

# gene selection
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)

# rebuild AnnData
adata_uni = sc.AnnData(X_df)
adata_uni.obs = adata.obs.copy()
adata_uni.uns = adata.uns.copy()

# basic filtering
adata_uni = BasicFilter(adata_uni, qc_min_genes=200, qc_min_cells=0)
adata_uni = QC_Metrics_info(adata_uni)

# save preprocessed AnnData
output_h5ad = os.path.join(output_dir, f"preprocessed_{file_name}")
print(f"[SCF-preprocess] Saving preprocessed h5ad to: {output_h5ad}")
adata_uni.write_h5ad(output_h5ad)

# save cell info
cell_ids = adata_uni.obs.index
try:
    batch_data = adata_uni.obs['Batch']
    df = pd.DataFrame({'Cell_ID': cell_ids, 'Batch': batch_data})
except KeyError:
    df = pd.DataFrame({'Cell_ID': cell_ids})
    print("[SCF-preprocess] No 'Batch' column in obs, skipping.")

output_excel = os.path.join(
    output_dir,
    f"preprocessed_{file_name.replace('.h5ad', '_info.xlsx')}"
)
df.to_excel(output_excel, index=False)

print("[SCF-preprocess] Done.")
