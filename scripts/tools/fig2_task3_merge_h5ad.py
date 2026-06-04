# merge_h5ad.py

import anndata as ad
import argparse
import sys
import os

def verify_merged_h5ad(file_path):
    """
     andafter .h5ad fileand keybatch infoto validate.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: validate , file '{file_path}' exist.")
        return
        
    print("\n--- [validation step] ---")
    print(f"INFO: in file '{file_path}' checks...")
    
    try:
        adata_check = ad.read_h5ad(file_path)
        
        print("\n1. andafter AnnData object must:")
        print(adata_check)
        
        print("\n2. check 'perturbation_status' cols:")
        if 'perturbation_status' in adata_check.obs.columns:
            print("INFO: 'perturbation_status' colsexist.")
            # statsand cols countamount
            status_counts = adata_check.obs['perturbation_status'].value_counts()
            print(" undercellcountamount:")
            print(status_counts)
        else:
            print("WARNING: found 'perturbation_status' cols!")

        print("\n3. cell data (obs) before5 :")
        print(adata_check.obs.head())
        
        print("\n4. cell data (obs) after5 (tocheck data):")
        print(adata_check.obs.tail())

        print(f"\nINFO: validate . data contain {adata_check.n_obs} cell {adata_check.n_vars} gene.")
        print("--- [validate ] ---\n")

    except Exception as e:
        print(f"ERROR: validateprocess orcheckfile '{file_path}' whenoccurred : {e}")


def merge_h5ad_files(control_path, perturbed_path, output_path):
    """
    merge twoh5adfile, andadd obscols .

    Args:
        control_path (str): .h5ad filepath.
        perturbed_path (str): .h5ad filepath.
        output_path (str): output andafter .h5ad file path.
    """
    try:
        # --- step 1: h5adfile ---
        print(f"INFO: Loading from '{control_path}' data...")
        adata_control = ad.read_h5ad(control_path)

        print(f"INFO: Loading from '{perturbed_path}' data...")
        adata_perturbed = ad.read_h5ad(perturbed_path)

        # --- step 2: add 'perturbation_status' cols ---
        adata_control.obs['perturbation_status'] = 'Control'
        print("INFO: as dataadd 'perturbation_status' = 'Control'.")

        adata_perturbed.obs['perturbation_status'] = 'IFN'
        print("INFO: as dataadd 'perturbation_status' = 'IFN'.")
        
        # --- step 3: merge twoAnnDataobject ---
        # AnnData.concatenate default using 'inner' join, willkeep
        # in dataobject existgene ( amount), expects as.
        print("INFO: inmerge two AnnData object...")
        adata_merged = adata_control.concatenate(
            adata_perturbed,
            join='inner' # canto 'inner' or 'outer'
        )

        print(f"INFO: anddone. data contain {adata_merged.n_obs} cell {adata_merged.n_vars} gene.")

        # --- step 4: andafterfile ---
        print(f"INFO: Writing andafterdata to '{output_path}'...")
        adata_merged.write_h5ad(output_path, compression="gzip")
        
        print("\n🎉 handledone!")

        # --- step 5: validate file ---
        verify_merged_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: file found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: handleprocess occurred - {e}")
        sys.exit(1)


if __name__ == '__main__':
    # --- argparse setup ---
    parser = argparse.ArgumentParser(
        description="merge two .h5ad file, andadd 'perturbation_status' to .",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'control_h5ad', 
        help='input .h5ad filepath.'
    )
    parser.add_argument(
        'perturbed_h5ad', 
        help='input .h5ad filepath.'
    )
    parser.add_argument(
        'output_h5ad', 
        help='output andafter .h5ad filepath.'
    )

    # parseCLIargs
    args = parser.parse_args()

    # call count and
    merge_h5ad_files(args.control_h5ad, args.perturbed_h5ad, args.output_h5ad)
