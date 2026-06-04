import sys
import os
import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

def get_label_from_filename(filepath):
    """
    based onfilename key specify labels.
    - filenamewith 'coculture' or 'ifn', as 'IFN'.
    - filenamewith 'control', as 'Control'.
    """
    # get filename, withpath name, key 
    filename_lower = os.path.basename(filepath).lower().replace('.h5ad', '')

    if 'coculture' in filename_lower:
        return 'IFN'
    elif 'ifn' in filename_lower:
        return 'IFN'
    elif 'control' in filename_lower:
        return 'Control'
    else:
        # nofoundspecifykey , and using usinglogic
        print(f" : infilename '{os.path.basename(filepath)}' found 'control', 'coculture', or 'ifn'.")
        print("will usedefaultlogic (filename after '_' after ) as .")
        label = filename_lower.split('_')[-1].capitalize()
        return label

def merge_and_split_data(file1_path, file2_path, output_train_path, output_test_path):
    """
    merge twoh5adfile, after 80/20 astrain set test set.
    """
    print("--- Start andand ---")

    # --- 1. Read H5AD ---
    try:
        print(f" file 1: {file1_path}")
        adata1 = ad.read_h5ad(file1_path)
        print(f" file 2: {file2_path}")
        adata2 = ad.read_h5ad(file2_path)
    except Exception as e:
        print(f" : filewhen - {e}")
        sys.exit(1)

    # --- 2. Auto labels in .obs ---
    label1 = get_label_from_filename(file1_path)
    label2 = get_label_from_filename(file2_path)
    print(f"\n--- ---")
    print(f"file1 '{os.path.basename(file1_path)}'  labelsspecifyas: '{label1}'")
    print(f"file2 '{os.path.basename(file2_path)}'  labelsspecifyas: '{label2}'")

    adata1.obs['perturbation_status'] = label1
    adata2.obs['perturbation_status'] = label2

    # --- 3. Merge AnnData objects ---
    print("\n--- in anddata ---")
    # add fill_value=0 args
    merged_adata = ad.concat([adata1, adata2], join='outer', fill_value=0)
    print(f" anddone, {merged_adata.n_obs} cell.")
    print(" andafter cellcountamount:")
    print(merged_adata.obs['perturbation_status'].value_counts())

    merged_adata.obs.rename(columns={'celltype': 'Cell.Type'}, inplace=True)
    
    # --- 4. Stratified 80/20 split ---
    print("\n--- in 80/20 data ---")

    indices = np.arange(merged_adata.n_obs)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=merged_adata.obs['perturbation_status']
    )

    adata_train = merged_adata[train_indices, :]
    adata_test = merged_adata[test_indices, :]

    # --- 5. Validate split ---
    print("\n--- validate results ---")
    print(f"train setshape: {adata_train.shape}")
    print("train set cellcountamount:")
    print(adata_train.obs['perturbation_status'].value_counts())

    print(f"\ntest setshape: {adata_test.shape}")
    print("test set cellcountamount:")
    print(adata_test.obs['perturbation_status'].value_counts())

    # --- 6. Save train/test ---
    try:
        print(f"\n--- Savingfile ---")
        adata_train.write_h5ad(output_train_path)
        print(f"train set to: '{output_train_path}'")
        adata_test.write_h5ad(output_test_path)
        print(f"test set to: '{output_test_path}'")
        print("\n✅ done!")
    except Exception as e:
        print(f" : filewhen - {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("\n[ ] wrong argument count!")
        print("using : python merge_and_split.py <inputfile1> <inputfile2> <outputtrain setpath> <outputtest setpath>")
        print("\n :")
        print("  python merge_and_split.py task4_ACTA2_control.h5ad task4_ACTA2_coculture.h5ad train_data.h5ad test_data.h5ad")
        sys.exit(1)

    file1, file2, out_train, out_test = sys.argv[1:5]
    merge_and_split_data(file1, file2, out_train, out_test)