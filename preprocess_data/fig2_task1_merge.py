import pandas as pd
import anndata as ad
import argparse # 导入 argparse 库

def main():
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description='将表达矩阵 (exp) 和元数据 (meta) CSV 文件合并成一个 H5AD 文件。')
    
    # 定义必需的参数
    parser.add_argument('--exp', type=str, required=True, help='输入的表达矩阵 CSV 文件路径 (cells x genes)。')
    parser.add_argument('--meta', type=str, required=True, help='输入的元数据 CSV 文件路径。')
    parser.add_argument('--output', type=str, required=True, help='输出的 H5AD 文件路径。')
    
    # 解析传入的参数
    args = parser.parse_args()

    # --- 2. 加载数据 ---
    print(f"正在从 {args.exp} 加载表达矩阵...")
    # index_col=0 表示使用第一列作为行索引 (细胞 ID)
    exp_df = pd.read_csv(args.exp, index_col=0)

    print(f"正在从 {args.meta} 加载元数据...")
    # 同样，使用第一列作为行索引
    meta_df = pd.read_csv(args.meta, index_col=0)
    
    if 'celltype' in meta_df.columns:
        meta_df.rename(columns={'celltype': 'Cell.Type'}, inplace=True)

    # --- 3. 创建 AnnData 对象 ---
    # AnnData 期望的输入是 (观测值 × 变量)，即 (细胞 × 基因)
    adata = ad.AnnData(exp_df)

    # --- 4. 匹配并添加元数据 ---
    print("正在匹配细胞 ID 并添加元数据...")
    # 为了确保安全，我们只保留那些同时存在于表达矩阵和元数据中的细胞
    # 并按照表达矩阵的顺序对元数据进行排序
    common_cells = exp_df.index.intersection(meta_df.index)
    adata = adata[common_cells, :].copy()
    adata.obs = meta_df.loc[common_cells]

    # --- 5. 保存为 h5ad 文件 ---
    print(f"正在将合并后的数据保存到 {args.output}...")
    adata.write_h5ad(args.output, compression='gzip')

    print("\n处理完成! 🎉")
    print("\n生成的 AnnData 对象信息如下:")
    print(adata)

if __name__ == '__main__':
    main()