import anndata as ad
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Merge control with train/test H5AD files and add perturbation_status labels.'
    )

    parser.add_argument('--control', type=str, required=True, help='Control H5AD path.')
    parser.add_argument('--train', type=str, required=True, help='Train H5AD path.')
    parser.add_argument('--test', type=str, required=True, help='Test H5AD path.')
    parser.add_argument('--output_train', type=str, required=True, help='Control + train merged output path.')
    parser.add_argument('--output_test', type=str, required=True, help='Control + test merged output path.')

    args = parser.parse_args()

    print("Loading H5AD files...")
    try:
        adata_control = ad.read_h5ad(args.control)
        print(f"Loaded control: {args.control}")
        adata_train = ad.read_h5ad(args.train)
        print(f"Loaded train: {args.train}")
        adata_test = ad.read_h5ad(args.test)
        print(f"Loaded test: {args.test}")
    except FileNotFoundError as e:
        print(f"File load error: {e}")
        return

    print("\nAdding 'perturbation_status' labels...")

    adata_control.obs['perturbation_status'] = 'Control'
    adata_train.obs['perturbation_status'] = 'IFN'
    adata_test.obs['perturbation_status'] = 'IFN'

    print("Labels added.")

    print("\nMerging control and train...")
    control_train_merged = ad.concat(
        {'control': adata_control, 'train': adata_train},
        join='inner',
        label='source'
    )

    print(f"Saving control + train to: {args.output_train}")
    control_train_merged.write_h5ad(args.output_train, compression='gzip')
    print("Control + train merge done.")

    print("\nMerging control and test...")
    control_test_merged = ad.concat(
        {'control': adata_control, 'test': adata_test},
        join='inner',
        label='source'
    )

    print(f"Saving control + test to: {args.output_test}")
    control_test_merged.write_h5ad(args.output_test, compression='gzip')
    print("Control + test merge done.")

    print("\nAll tasks finished.")
    print("\nMerged file summary:")
    print(f"  {args.output_train}: {control_train_merged.n_obs} cells, {control_train_merged.n_vars} genes")
    print(f"  {args.output_test}: {control_test_merged.n_obs} cells, {control_test_merged.n_vars} genes")

    print("\nFinal obs columns:")
    print(control_train_merged.obs.columns.tolist())


if __name__ == '__main__':
    main()
