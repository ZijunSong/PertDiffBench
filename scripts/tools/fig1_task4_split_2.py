import sys
import os
import anndata as ad
import pandas as pd

# and preprocess_data/fig1/create_global_gene_list.py, fig1_task4_merge.sh 
DATA_DIR = "/data/ppnm/data/PertDiffBench/data/fig1_task4"
GLOBAL_GENE_LIST_PATH = os.path.join(DATA_DIR, "global_gene_list.txt")

# --- global gene list ---
try:
    with open(GLOBAL_GENE_LIST_PATH, "r") as f:
        GLOBAL_GENE_LIST = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"❌ : global gene list not found: {GLOBAL_GENE_LIST_PATH}")
    print("   run first:: python preprocess_data/fig1/create_global_gene_list.py")
    sys.exit(1)


def merge_pair(file1_path, label1, file2_path, label2, output_path):
    # ... ( countinside otherlogic ) ...
    print("-" * 50)
    print(f"handle : {os.path.basename(file1_path)} + {os.path.basename(file2_path)} -> {os.path.basename(output_path)}")
    
    adata1 = ad.read_h5ad(file1_path)
    adata2 = ad.read_h5ad(file2_path)
    adata1.obs['perturbation_status'] = label1
    adata2.obs['perturbation_status'] = label2
    merged_adata = ad.concat([adata1, adata2], join='outer', fill_value=0)

    # ✅ : usingglobal gene listaligndata
    print(f"  - shape after merge: {merged_adata.shape}")
    print(f"  - aligning to {len(GLOBAL_GENE_LIST)}  global genes...")
    merged_adata = ad.AnnData(merged_adata.to_df().reindex(columns=GLOBAL_GENE_LIST, fill_value=0), obs=merged_adata.obs)
    print(f"  - alignaftershape: {merged_adata.shape}")

    if 'celltype' in merged_adata.obs.columns:
        merged_adata.obs.rename(columns={'celltype': 'Cell.Type'}, inplace=True)
        print(" - 'celltype' cols nameas 'Cell.Type'.")

    merged_adata.write_h5ad(output_path)
    print(f" ✅ !saved to: {output_path}")
    print("-" * 50)


if __name__ == '__main__':
    # ... ( countlogic ) ...
    if len(sys.argv) != 6:
        print("\n[ ] wrong argument count!")
        print("\nusing : python create_merged_datasets.py <controlfile> <coculturefile> <ifnfile> <outputcontrol-coculturefile> <outputcontrol-ifnfile>")
        sys.exit(1)

    control_file, coculture_file, ifn_file, output_coculture_path, output_ifn_path = sys.argv[1:6]
    
    merge_pair(control_file, 'Control', coculture_file, 'IFN', output_coculture_path)
    merge_pair(control_file, 'Control', ifn_file, 'IFN', output_ifn_path)