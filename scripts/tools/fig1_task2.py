import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

data_dir = 'data/fig1/task2/'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"Error: No CSV files found in {data_dir}.")
    exit()

for csv_file in csv_files:
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(data_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- Processing file: {csv_file} ---")

    try:
        df = pd.read_csv(csv_file, index_col=0)
        
        # 查看CSV数据的详细信息
        print(f"--- Details for {base_name} ---")
        print("DataFrame Info:")
        df.info()
        print("\nDataFrame Head:")
        print(df.head())
        print("-" * (20 + len(base_name)))

    except FileNotFoundError:
        print(f"Error: File not found {csv_file}. Please check the path.")
        continue
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        continue

    try:
        cell_ids = df.columns.tolist()
        dose_value_unit_data = df.loc['dose_value_unit', :].values
        treatment_agent_data = df.loc['treatment_agent', :].values
    except KeyError as e:
        print(f"Error: Metadata row {e} not found. Please check if 'dose_value_unit' and 'treatment_agent' exist as row indices in the CSV.")
        continue

    df_expression = df.drop(index=['dose_value_unit', 'treatment_agent'], errors='ignore')
    
    df_expression = df_expression.apply(pd.to_numeric, errors='coerce')
    if df_expression.isnull().any().any():
        print("Warning: Non-numeric content found in expression data. Coerced to NaN and filled with 0.")
        df_expression = df_expression.fillna(0)

    df_transposed = df_expression.T
    
    # *** 修正 ***
    # 直接将DataFrame传递给AnnData，以保留细胞ID（索引）和基因ID（列名）
    adata = ad.AnnData(df_transposed, dtype='float32')
    
    adata.obs['dose_value_unit'] = dose_value_unit_data
    adata.obs['treatment_agent'] = treatment_agent_data
    
    try:
        cell_type = base_name.split('_')[3]
        adata.obs['Cell.Type'] = cell_type
    except IndexError:
        print("Warning: Could not parse 'Cell.Type' from filename. Using 'unknown'.")
        adata.obs['Cell.Type'] = 'unknown'

    conditions = [
        adata.obs['treatment_agent'] == 'DMSO',
        adata.obs['treatment_agent'] == 'aflatoxin B1'
    ]
    choices = ['Control', 'IFN']
    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(f"\nFinal expression data matrix shape (cells x genes): {adata.X.shape}")

    try:
        adata.write(h5ad_file)
        print(f"File successfully saved to: {h5ad_file}")
    except Exception as e:
        print(f"Error saving H5AD file: {e}")

print("\n------ All file conversions complete ------")
