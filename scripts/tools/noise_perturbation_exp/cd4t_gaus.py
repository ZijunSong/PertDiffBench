import anndata
import numpy as np
import scipy.sparse as sp
import os

def generate_noisy_files(file_path, output_dir):
    """
     .h5ad file, to expressedmatrixadd types noise, 
    andaseach results as .h5ad file.

    args:
    file_path (str): input .h5ad filepath.
    output_dir (str): alloutputfilefolderpath.
    """
    try:
        # 1. Ensureoutputfolderexist
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- Output directory: {output_dir} ---")

        # 2. h5ad file
        print(f"--- Loadingdata: {file_path} ---")
        adata_original = anndata.read_h5ad(file_path)
        
        print("\n--- originaldata under ---")
        print(adata_original)
        
        is_sparse = sp.issparse(adata_original.X)
        print(f"\nexpressedmatrix (adata.X) whetherassparse matrix: {is_sparse}")
        
        # ashandlesparse matrix, we convert to array
        # dataamount , will amountinside 
        original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()

        # 3. define noise ( )
        noise_levels = [0.1, 0.25, 0.5, 1.0, 1.5]
        print(f"\n--- mustaddnoise ( ): {noise_levels} ---")

        # getoriginalfilename , for name file
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # 4. addnoiseand as file
        for scale in noise_levels:
            print(f"\nProcessingnoiselevel (scale={scale})...")
            
            # noise
            noise = np.random.normal(loc=0, scale=scale, size=original_data.shape)
            
            # noiseaddtooriginaldata
            noisy_data = original_data + noise
            
            # allnegative values as0
            noisy_data[noisy_data < 0] = 0
            
            # AnnData object noisedata
            # whenkeeporiginalcell(obs) gene(var) 
            adata_noisy = anndata.AnnData(noisy_data, obs=adata_original.obs, var=adata_original.var)
            
            # filename
            output_filename = f"{base_filename}_noise_std_{scale}.h5ad"
            output_path = os.path.join(output_dir, output_filename)
            
            # as .h5ad file
            print(f"Saving to: {output_path}")
            adata_noisy.write_h5ad(output_path)
        
        print("\n--- all noisefile ! ---")

    except FileNotFoundError:
        print(f" : file found, checkpath '{file_path}' whethercorrect.")
    except Exception as e:
        print(f"handleprocess occurred : {e}")

# --- using ---
if __name__ == '__main__':
    # under path as filepath
    input_file = '/share/PertBench/data/fig1/raw_task1/task1_valid_CD4T_exp.h5ad' 
    
    # h5ad file runtest
    if not os.path.exists(input_file):
        print(f" found '{input_file}'.Creating for testing h5ad file...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.rand(n_obs, n_vars) * 10
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f" file '{input_file}'.")

    # specify alloutputfilefolder
    output_directory = '/share/PertBench/data/add_gaussian_noise_output'
    
    generate_noisy_files(input_file, output_directory)
