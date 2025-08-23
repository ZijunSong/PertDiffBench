import anndata
import numpy as np
import scipy.sparse as sp
import os

def generate_noisy_files(file_path, output_dir):
    """
    加载一个 .h5ad 文件，向其表达矩阵添加五种不同梯度的高斯噪声，
    并为每个梯度的结果分别保存为新的 .h5ad 文件。

    参数:
    file_path (str): 输入的 .h5ad 文件路径。
    output_dir (str): 保存所有输出文件的文件夹路径。
    """
    try:
        # 1. 确保输出文件夹存在
        os.makedirs(output_dir, exist_ok=True)
        print(f"--- 输出文件将保存在: {output_dir} ---")

        # 2. 加载 h5ad 文件
        print(f"--- 正在加载数据: {file_path} ---")
        adata_original = anndata.read_h5ad(file_path)
        
        print("\n--- 原始数据格式如下 ---")
        print(adata_original)
        
        is_sparse = sp.issparse(adata_original.X)
        print(f"\n表达矩阵 (adata.X) 是否为稀疏矩阵: {is_sparse}")
        
        # 为了处理稀疏矩阵，我们先将其转换为密集数组
        # 如果数据量非常大，请注意这会消耗大量内存
        original_data = adata_original.X.toarray() if is_sparse else adata_original.X.copy()

        # 3. 定义五个噪声梯度（标准差）
        noise_levels = [0.1, 0.25, 0.5, 1.0, 1.5]
        print(f"\n--- 将要添加的噪声梯度 (标准差): {noise_levels} ---")

        # 获取原始文件名的基本部分，用于命名新文件
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # 4. 循环添加噪声并分别保存为新文件
        for scale in noise_levels:
            print(f"\n正在处理噪声级别 (scale={scale})...")
            
            # 生成高斯噪声
            noise = np.random.normal(loc=0, scale=scale, size=original_data.shape)
            
            # 将噪声添加到原始数据
            noisy_data = original_data + noise
            
            # 将所有负值裁剪为0
            noisy_data[noisy_data < 0] = 0
            
            # 创建一个新的 AnnData 对象来存储带噪声的数据
            # 同时保留原始的细胞(obs)和基因(var)注释
            adata_noisy = anndata.AnnData(noisy_data, obs=adata_original.obs, var=adata_original.var)
            
            # 构建新的文件名
            output_filename = f"{base_filename}_noise_std_{scale}.h5ad"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存为新的 .h5ad 文件
            print(f"正在保存到: {output_path}")
            adata_noisy.write_h5ad(output_path)
        
        print("\n--- 所有带噪声的文件均已成功生成！ ---")

    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径 '{file_path}' 是否正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 使用示例 ---
if __name__ == '__main__':
    # 请将下面的路径替换为您的实际文件路径
    input_file = '/share/PertBench/data/fig1/task1/task1_valid_CD4T_exp.h5ad' 
    
    # 创建一个虚拟的 h5ad 文件来进行测试
    if not os.path.exists(input_file):
        print(f"未找到 '{input_file}'。正在创建一个用于测试的虚拟 h5ad 文件...")
        n_obs, n_vars = 100, 500
        X_dummy = np.random.rand(n_obs, n_vars) * 10
        dummy_adata = anndata.AnnData(X_dummy)
        dummy_adata.write(input_file)
        print(f"已创建虚拟文件 '{input_file}'。")

    # 指定保存所有输出文件的文件夹
    output_directory = '/share/PertBench/data/add_gaussian_noise_output'
    
    generate_noisy_files(input_file, output_directory)
