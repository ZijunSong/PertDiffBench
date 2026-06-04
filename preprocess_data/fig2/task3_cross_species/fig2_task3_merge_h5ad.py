# merge_h5ad.py

import anndata as ad
import argparse
import sys
import os

def verify_merged_h5ad(file_path):
    """Read a merged .h5ad file and print key info for validation."""
    if not os.path.exists(file_path):
        print(f"ERROR: validation failed, file '{file_path}' does not exist.")
        return

    print("\n--- [validation] ---")
    print(f"INFO: re-reading saved file '{file_path}' for checks...")

    try:
        adata_check = ad.read_h5ad(file_path)

        print("\n1. Merged AnnData summary:")
        print(adata_check)

        print("\n2. check 'perturbation_status' column:")
        if 'perturbation_status' in adata_check.obs.columns:
            print("INFO: 'perturbation_status' column exists.")
            status_counts = adata_check.obs['perturbation_status'].value_counts()
            print("Cell counts per status:")
            print(status_counts)
        else:
            print("WARNING: 'perturbation_status' column not found.")

        print("\n3. cell metadata (obs) first 5 rows:")
        print(adata_check.obs.head())

        print("\n4. cell metadata (obs) last 5 rows:")
        print(adata_check.obs.tail())

        print(f"\nINFO: validation OK. Final dataset: {adata_check.n_obs} cells, {adata_check.n_vars} genes.")
        print("--- [validation end] ---\n")

    except Exception as e:
        print(f"ERROR: validation failed reading '{file_path}': {e}")


def merge_h5ad_files(control_path, perturbed_path, output_path):
    """
    Merge two h5ad files and add obs column to distinguish them.

    Args:
        control_path (str): control group .h5ad path.
        perturbed_path (str): perturbed group .h5ad path.
        output_path (str): merged .h5ad output path.
    """
    try:
        print(f"INFO: loading control from '{control_path}'...")
        adata_control = ad.read_h5ad(control_path)

        print(f"INFO: loading perturbed from '{perturbed_path}'...")
        adata_perturbed = ad.read_h5ad(perturbed_path)

        adata_control.obs['perturbation_status'] = 'Control'
        print("INFO: set perturbation_status='Control' on control data.")

        adata_perturbed.obs['perturbation_status'] = 'IFN'
        print("INFO: set perturbation_status='IFN' on perturbed data.")

        print("INFO: concatenating AnnData objects...")
        adata_merged = adata_control.concatenate(
            adata_perturbed,
            join='inner'
        )

        print(f"INFO: merge done. New dataset: {adata_merged.n_obs} cells, {adata_merged.n_vars} genes.")

        print(f"INFO: saving merged data to '{output_path}'...")
        adata_merged.write_h5ad(output_path, compression="gzip")

        print("\nDone.")

        verify_merged_h5ad(output_path)

    except FileNotFoundError as e:
        print(f"ERROR: file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: processing failed - {e}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge two .h5ad files and add 'perturbation_status' labels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('control_h5ad', help='Control group .h5ad path.')
    parser.add_argument('perturbed_h5ad', help='Perturbed group .h5ad path.')
    parser.add_argument('output_h5ad', help='Merged .h5ad output path.')

    args = parser.parse_args()
    merge_h5ad_files(args.control_h5ad, args.perturbed_h5ad, args.output_h5ad)
