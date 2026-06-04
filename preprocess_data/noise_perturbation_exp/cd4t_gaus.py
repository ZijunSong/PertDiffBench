import anndata
import numpy as np
import scipy.sparse as sp
import os

def generate_noisy_files(file_path, output_dir):
    """
    Load one .h5ad file, add five levels of Gaussian noise to its expression matrix,
    and save each result as a separate .h5ad file.

    Args:
        file_path (str): Input .h5ad file path.
        output_dir (str): Directory path for all output files.
    """
    try:
        # 1. Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Output directory ready: {output_dir} ---")

        # 2. Load h5ad file
        print(f"--- Loading data: {file_path} ---")
        adata_original = anndata.read_h5ad(file_path)

        print("\n--- Original data layout ---")
        print(adata_original)

        is_sparse = sp.issparse(adata_original.X)
        print(f"\nexpression matrix (adata.X) is sparse: {is_sparse}")

        # For sparse matrices, convert to dense array first
        # NOTE: Very large data may use substantial memory
        original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()

        # 3. Define five noise levels (standard deviation)
        noise_levels = [0.1, 0.25, 0.5, 1.0, 1.5]
        print(f"\n--- Noise levels (std): {noise_levels} ---")

        # Base filename for naming output files
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # 4. Loop: add noise and save each file
        for scale in noise_levels:
            print(f"\nProcessing noise level (scale={scale})...")

            # Generate Gaussian noise
            noise = np.random.normal(loc=0, scale=scale, size=original_data.shape)

            # Add noise to original data
            noisy_data = original_data + noise

            # Clip negative values to 0
            noisy_data[noisy_data < 0] = 0

            # New AnnData for noisy data; keep original obs/var annotations
            adata_noisy = anndata.AnnData(noisy_data, obs=adata_original.obs, var=adata_original.var)

            # Build output filename
            output_filename = f"{base_filename}_noise_std_{scale}.h5ad"
            output_path = os.path.join(output_dir, output_filename)

            # Save new .h5ad file
            print(f"Saving to: {output_path}")
            adata_noisy.write_h5ad(output_path)

        print("\n--- All noisy files generated successfully ---")

    except FileNotFoundError:
        print(f"Error: file not found; check path '{file_path}'.")
    except Exception as e:
        print(f"Error during processing: {e}")

# --- Example usage ---
if __name__ == '__main__':
    # Replace with your actual file path
    input_file = '/share/PertBench/data/fig1/raw_task1/task1_valid_CD4T_exp.h5ad'

    # Create a dummy h5ad for testing if missing
    if not os.path.exists(input_file):
        print(f"Not found '{input_file}'. Creating dummy h5ad for testing...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.rand(n_obs, n_vars) * 10
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f"Created dummy file '{input_file}'.")

    # Output directory for all noisy files
    output_directory = '/share/PertBench/data/add_gaussian_noise_output'

    generate_noisy_files(input_file, output_directory)
