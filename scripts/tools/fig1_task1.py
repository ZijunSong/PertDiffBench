import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

# definefileindirectory
data_dir = 'data/fig1/raw_task1/'

print(f"Starthandledirectory CSVfile: {data_dir}...")

# getallCSVfile
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f" : in {data_dir} nofound CSVfile.")
    exit()

for csv_file in csv_files:
    # based onCSVfilename forshouldH5ADfilename
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(data_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- Processingfile: {csv_file} ---")

    # CSVfile
    try:
        df = pd.read_csv(csv_file, index_col=0)
        print("CSVfile .")
        print(f"data (cell x gene): {df.shape}")
    except FileNotFoundError:
        print(f" : tofile {csv_file}. checkpathwhethercorrect.")
        continue # continue to next file
    except Exception as e:
        print(f" CSVfile {csv_file} failed with: {e}")
        continue # continue to next file

    # AnnDataobject
    adata = ad.AnnData(df)
    print("AnnDataobject .")
    # print(adata) # Uncomment for verbose details if needed

    # asAnnDataobjectadd data (adata.obs)
    print(" inadd datato adata.obs ...")

    # fromfilename 'Cell.Type'
    try:
        cell_type = base_name.split('_')[2]
        adata.obs['Cell.Type'] = cell_type
        print(f" add 'Cell.Type' cols: {cell_type}.")
    except IndexError:
        print(" : cannotfromfilename parse 'Cell.Type'.will use 'unknown'.")
        adata.obs['Cell.Type'] = 'unknown'


    # based onindex(cellID)suffix, add 'perturbation_status' cols
    conditions = [
        adata.obs.index.str.endswith('stimulated'),
        adata.obs.index.str.endswith('control')
    ]
    choices = ['IFN', 'Control']

    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')
    print(" add 'perturbation_status' cols.")

    # check afterAnnDataobject
    print(" dataadddone.")
    # print(adata) # Uncomment for verbose details if needed

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    # Print the first 5 rows of gene metadata (var)
    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(adata.X)

    # AnnDataobject H5ADfile
    try:
        adata.write(h5ad_file) # using .write() 
        print(f"file (contain data)as: {h5ad_file}")
    except Exception as e:
        print(f" H5ADfilefailed with: {e}")

    # reloadandvalidatefile (optional, file , canto skip to when )
    print("--- validation step ---")
    print(f" inreload H5ADfile: {h5ad_file}")
    try:
        adata_loaded = ad.read_h5ad(h5ad_file)
        print("filereload . objectbatch info:")
        # print(adata_loaded) # Uncomment for verbose details if needed
        print("validate after data (before5 ):")
        print(adata_loaded.obs.head())
    except Exception as e:
        print(f"reloadH5ADfile {h5ad_file} failed with: {e}")

print("\n------ allfileconvertdone ------")