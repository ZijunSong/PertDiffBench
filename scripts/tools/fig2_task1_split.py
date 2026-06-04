import anndata as ad
import argparse

def main():
    # --- 1. argparse setup ---
    parser = argparse.ArgumentParser(
        description=' and control H5AD file and train test H5AD file, andadd perturbation_status .'
    )
    
    # defineinputfileargs
    parser.add_argument('--control', type=str, required=True, help='input control H5AD filepath.')
    parser.add_argument('--train', type=str, required=True, help='input train H5AD filepath.')
    parser.add_argument('--test', type=str, required=True, help='input test H5AD filepath.')
    
    # defineoutputfileargs
    parser.add_argument('--output_train', type=str, required=True, help='Control + Train andafteroutputfilepath.')
    parser.add_argument('--output_test', type=str, required=True, help='Control + Test andafteroutputfilepath.')
    
    args = parser.parse_args()

    # --- 2. Load all data ---
    print("🚀 Start H5AD file...")
    try:
        adata_control = ad.read_h5ad(args.control)
        print(f"✅ Control file: {args.control}")
        adata_train = ad.read_h5ad(args.train)
        print(f"✅ Train file: {args.train}")
        adata_test = ad.read_h5ad(args.test)
        print(f"✅ Test file: {args.test}")
    except FileNotFoundError as e:
        print(f"❌ file : {e}")
        return

    # --- 3. Add perturbation_status before merge ---
    print("\n🏷️ Runningeachdata add 'perturbation_status' ...")
    
    # as control data value 'Control'
    adata_control.obs['perturbation_status'] = 'Control'
    
    # as train test data value 'IFN'
    adata_train.obs['perturbation_status'] = 'IFN'
    adata_test.obs['perturbation_status'] = 'IFN'
    
    print("👍 adddone!")

    # --- 4. Merge Control + Train ---
    print("\n🔗 in and Control Train data...")
    
    # using 'source' cols, will data 
    control_train_merged = ad.concat(
        {'control': adata_control, 'train': adata_train},
        join='inner',
        label='source' # 'source' colswill cell 'control' 'train'
    )
    
    print(f"📝 Saving Control + Train andfileto: {args.output_train}")
    control_train_merged.write_h5ad(args.output_train, compression='gzip')
    print("👍 Control + Train anddone!")

    # --- 5. and Control + Test ---
    print("\n🔗 in and Control Test data...")

    control_test_merged = ad.concat(
        {'control': adata_control, 'test': adata_test},
        join='inner',
        label='source' # 'source' colswill cell 'control' 'test'
    )

    print(f"📝 Saving Control + Test andfileto: {args.output_test}")
    control_test_merged.write_h5ad(args.output_test, compression='gzip')
    print("👍 Control + Test anddone!")

    print("\n\n🎉 all handle !")
    print("\n andafterfile must:")
    print(f"👉 {args.output_train}: {control_train_merged.n_obs} cell, {control_train_merged.n_vars} gene")
    print(f"👉 {args.output_test}: {control_test_merged.n_obs} cell, {control_test_merged.n_vars} gene")
    
    # datacolsnameto check
    print("\n datacontaincols:")
    print(control_train_merged.obs.columns.tolist())


if __name__ == '__main__':
    main()