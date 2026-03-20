import anndata as ad
import argparse

def main():
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description='合并 control H5AD 文件分别与 train 和 test H5AD 文件，并添加 perturbation_status 标签。'
    )
    
    # 定义输入文件参数
    parser.add_argument('--control', type=str, required=True, help='输入的 control H5AD 文件路径。')
    parser.add_argument('--train', type=str, required=True, help='输入的 train H5AD 文件路径。')
    parser.add_argument('--test', type=str, required=True, help='输入的 test H5AD 文件路径。')
    
    # 定义输出文件参数
    parser.add_argument('--output_train', type=str, required=True, help='Control + Train 合并后的输出文件路径。')
    parser.add_argument('--output_test', type=str, required=True, help='Control + Test 合并后的输出文件路径。')
    
    args = parser.parse_args()

    # --- 2. 加载所有数据 ---
    print("🚀 开始加载 H5AD 文件...")
    try:
        adata_control = ad.read_h5ad(args.control)
        print(f"✅ 成功加载 Control 文件: {args.control}")
        adata_train = ad.read_h5ad(args.train)
        print(f"✅ 成功加载 Train 文件: {args.train}")
        adata_test = ad.read_h5ad(args.test)
        print(f"✅ 成功加载 Test 文件: {args.test}")
    except FileNotFoundError as e:
        print(f"❌ 文件加载错误: {e}")
        return

    # --- 3. 【新增】在合并前添加 perturbation_status 列 ---
    print("\n🏷️ 正在为每个数据集添加 'perturbation_status' 标签...")
    
    # 为 control 数据集赋值 'Control'
    adata_control.obs['perturbation_status'] = 'Control'
    
    # 为 train 和 test 数据集赋值 'IFN'
    adata_train.obs['perturbation_status'] = 'IFN'
    adata_test.obs['perturbation_status'] = 'IFN'
    
    print("👍 标签添加完成!")

    # --- 4. 合并 Control + Train ---
    print("\n🔗 正在合并 Control 和 Train 数据...")
    
    # 使用字典的键来创建 'source' 列，这会自动标记数据来源
    control_train_merged = ad.concat(
        {'control': adata_control, 'train': adata_train},
        join='inner',
        label='source' # 'source' 列会标记细胞来自 'control' 还是 'train'
    )
    
    print(f"📝 正在保存 Control + Train 合并文件到: {args.output_train}")
    control_train_merged.write_h5ad(args.output_train, compression='gzip')
    print("👍 Control + Train 合并完成!")

    # --- 5. 合并 Control + Test ---
    print("\n🔗 正在合并 Control 和 Test 数据...")

    control_test_merged = ad.concat(
        {'control': adata_control, 'test': adata_test},
        join='inner',
        label='source' # 'source' 列会标记细胞来自 'control' 还是 'test'
    )

    print(f"📝 正在保存 Control + Test 合并文件到: {args.output_test}")
    control_test_merged.write_h5ad(args.output_test, compression='gzip')
    print("👍 Control + Test 合并完成!")

    print("\n\n🎉 所有任务处理完毕!")
    print("\n合并后文件摘要:")
    print(f"👉 {args.output_train}: {control_train_merged.n_obs} 个细胞, {control_train_merged.n_vars} 个基因")
    print(f"👉 {args.output_test}: {control_test_merged.n_obs} 个细胞, {control_test_merged.n_vars} 个基因")
    
    # 打印最终元数据列名以供检查
    print("\n最终元数据包含的列:")
    print(control_train_merged.obs.columns.tolist())


if __name__ == '__main__':
    main()