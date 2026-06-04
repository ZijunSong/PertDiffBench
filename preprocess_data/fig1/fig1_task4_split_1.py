import sys
import os
import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

def get_label_from_filename(filepath):
    """
    Assign labels from filename keywords.
 - If filename contains 'coculture' or 'ifn', label is 'IFN'.
 - If filename contains 'control', label is 'Control'.
    """
    # Lowercase basename for keyword matching
    filename_lower = os.path.basename(filepath).lower().replace('.h5ad', '')

    if 'coculture' in filename_lower:
        return 'IFN'
    elif 'ifn' in filename_lower:
        return 'IFN'
    elif 'control' in filename_lower:
        return 'Control'
    else:
        # Fallback: warn and use suffix after last '_'
        print(f"Warning:  in filenames '{os.path.basename(filepath)}' innot found 'control', 'coculture',  or  'ifn'.")
        print("Using default workflow (suffix after last '_') as label.")
        label = filename_lower.split('_')[-1].capitalize()
        return label

def merge_and_split_data(file1_path, file2_path, output_train_path, output_test_path):
    """
    Merge two h5ad files, then stratified 80/20 train/test split.
    """
    print("--- Starting merge and split ---")

    # --- 1. read H5AD file ---
    try:
        print(f"readfile 1: {file1_path}")
        adata1 = ad.read_h5ad(file1_path)
        print(f"readfile 2: {file2_path}")
        adata2 = ad.read_h5ad(file2_path)
    except Exception as e:
        print(f"Error: failed to read files - {e}")
        sys.exit(1)

    # --- 2. Auto-generate labels and add to .obs ---
    label1 = get_label_from_filename(file1_path)
    label2 = get_label_from_filename(file2_path)
    print("\n--- Auto-generated labels ---")
    print(f"file1 '{os.path.basename(file1_path)}' label: '{label1}'")
    print(f"file2 '{os.path.basename(file2_path)}' label: '{label2}'")

    adata1.obs['perturbation_status'] = label1
    adata2.obs['perturbation_status'] = label2

    # --- 3. Merge two AnnData objects ---
    print("\n--- Merging data ---")
    # add  fill_value=0 param 
    merged_adata = ad.concat([adata1, adata2], join='outer', fill_value=0)
    print(f"Merge done, total cells: {merged_adata.n_obs}")
    print("Cell counts per label after merge:")
    print(merged_adata.obs['perturbation_status'].value_counts())

    merged_adata.obs.rename(columns={'celltype': 'Cell.Type'}, inplace=True)
    
    # --- 4. Stratified 80/20 split ---
    print("\n--- 80/20 stratified split ---")

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
    print("\n--- Split validation ---")
    print(f"train set shape: {adata_train.shape}")
    print("Train set cells per label:")
    print(adata_train.obs['perturbation_status'].value_counts())

    print(f"\ntest set shape: {adata_test.shape}")
    print("Test set cells per label:")
    print(adata_test.obs['perturbation_status'].value_counts())

    # --- 6. Save train and test sets ---
    try:
        print(f"\n--- currently savefile ---")
        adata_train.write_h5ad(output_train_path)
        print(f"Train set saved to: '{output_train_path}'")
        adata_test.write_h5ad(output_test_path)
        print(f"Test set saved to: '{output_test_path}'")
        print("\nDone.")
    except Exception as e:
        print(f"Error: failed to save - {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("\n[Error] Wrong number of arguments.")
        print("Usage: python merge_and_split.py <input1> <input2> <train_out> <test_out>")
        print("\nexample:")
        print("  python merge_and_split.py task4_ACTA2_control.h5ad task4_ACTA2_coculture.h5ad train_data.h5ad test_data.h5ad")
        sys.exit(1)

    file1, file2, out_train, out_test = sys.argv[1:5]
    merge_and_split_data(file1, file2, out_train, out_test)