import pandas as pd
import anndata as ad
import numpy as np
import os
import glob

# 定义文件所在的目录
data_dir = 'data/fig1/task3/'

print(f"开始处理目录中的CSV文件: {data_dir}...")

# 获取所有CSV文件
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print(f"错误：在 {data_dir} 中没有找到任何CSV文件。")
    exit()

for csv_file in csv_files:
    # 根据CSV文件名生成对应的H5AD文件名
    base_name = os.path.basename(csv_file)
    h5ad_file = os.path.join(data_dir, base_name.replace('.csv', '.h5ad'))

    print(f"\n--- 正在处理文件: {csv_file} ---")

    # 读取CSV文件
    try:
        # *** 关键修改：读取CSV文件，跳过第一列（无用行号），将第二列 'index' 作为DataFrame的索引 ***
        # 这会使得DataFrame的行索引直接是细胞ID
        df = pd.read_csv(csv_file, index_col=1)
        print("CSV文件读取成功。")
        print(f"原始数据维度 (细胞 x 基因): {df.shape}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file}。请检查路径是否正确。")
        continue # 继续处理下一个文件
    except Exception as e:
        print(f"读取CSV文件 {csv_file} 时出错: {e}")
        continue # 继续处理下一个文件

    # *** 移除旧的元数据提取和转置逻辑 ***
    # 新数据已经是细胞 x 基因格式，且没有 'dose_value_unit' 和 'treatment_agent' 行
    # 所以直接使用读取的df作为AnnData的输入
    df_expression = df
    print("数据已确认是细胞 x 基因格式，无需转置。")

    # 创建AnnData对象
    # 此时 df_expression 的行是细胞，列是基因
    adata = ad.AnnData(df_expression)
    print("AnnData对象初步创建成功。")
    # print(adata) # 可以根据需要取消注释，查看详细信息

    # 为AnnData对象添加元数据 (adata.obs)
    # adata.obs.index 已经包含了细胞ID
    print("正在添加元数据到 adata.obs ...")

    # 从文件名中提取 'Cell.Type' (保持不变，如果文件名格式依然适用)
    try:
        cell_type = base_name.split('_')[2]
        adata.obs['Cell.Type'] = cell_type
        print(f"已添加 'Cell.Type' 列: {cell_type}。")
    except IndexError:
        print("警告：无法从文件名中解析 'Cell.Type'。将使用 'unknown'。")
        adata.obs['Cell.Type'] = 'unknown'

    # *** 关键修改：根据 adata.obs.index 设置 'perturbation_status' ***
    # 检查细胞ID（adata.obs.index）是否以 'control' 或 'stimulated' 结尾
    conditions = [
        adata.obs.index.str.endswith('control'),
        adata.obs.index.str.endswith('stimulated')
    ]
    choices = ['Control', 'IFN'] # 对应 control -> Control, stimulated -> IFN

    adata.obs['perturbation_status'] = np.select(conditions, choices, default='unknown')
    print("已根据细胞ID后缀添加 'perturbation_status' 列。")

    # 检查更新后的AnnData对象
    print("元数据添加完成。")
    # print(adata) # 可以根据需要取消注释，查看详细信息

    print("\n--- Cell Metadata (adata.obs, first 5 cells) ---")
    print(adata.obs.head())

    # Print the first 5 rows of gene metadata (var)
    print("\n--- Gene Metadata (adata.var, first 5 genes) ---")
    print(adata.var.head())

    print(adata.X)

    # 将AnnData对象写入H5AD文件
    try:
        adata.write(h5ad_file) # 使用 .write() 方法
        print(f"文件已成功保存（包含元数据）为: {h5ad_file}")
    except Exception as e:
        print(f"保存H5AD文件时出错: {e}")

    # 读回并验证文件 (可选，如果文件很多，可以考虑跳过此步以节省时间)
    print("--- 验证步骤 ---")
    print(f"正在读回已保存的H5AD文件: {h5ad_file}")
    try:
        adata_loaded = ad.read_h5ad(h5ad_file)
        print("文件读回成功。加载的对象信息:")
        # print(adata_loaded) # 可以根据需要取消注释，查看详细信息
        print("验证加载后的元数据 (前5行):")
        print(adata_loaded.obs.head())
        print("验证加载后的基因信息 (adata.var 的前5行):")
        print(adata_loaded.var.head()) # 检查基因信息
    except Exception as e:
        print(f"读回H5AD文件 {h5ad_file} 时出错: {e}")

print("\n------ 所有文件转换完成 ------")
