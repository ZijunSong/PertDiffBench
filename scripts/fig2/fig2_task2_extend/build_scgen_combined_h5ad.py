#!/usr/bin/env python3
"""
构建 scGen setting 下的训练用 h5ad：训练集 (Control+IFN) + 各测试集的 Control 细胞，
并添加 obs['split']：'train' 表示来自训练集（用于配对），'test_control' 表示来自测试集 Control（不参与配对）。
输出保存在 DATA_DIR 下：scgen_combined_train_plus_test_control.h5ad
"""
import os
import argparse
import scanpy as sc

def main():
    parser = argparse.ArgumentParser(description="Build combined train + test Control h5ad for scGen setting.")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Directory containing task1_train_CD4T_exp.h5ad and task2_test_*_exp.h5ad")
    parser.add_argument("--out", type=str, default=None,
                        help="Output h5ad path; default: <data_dir>/scgen_combined_train_plus_test_control.h5ad")
    args = parser.parse_args()
    if not args.data_dir:
        # 默认：项目根下 data/fig2/task2_unseen_celltype
        proj = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        data_dir = os.path.join(proj, "data", "fig2", "task2_unseen_celltype")
    else:
        data_dir = os.path.abspath(args.data_dir)
    out_path = args.out or os.path.join(data_dir, "scgen_combined_train_plus_test_control.h5ad")

    train_path = os.path.join(data_dir, "task1_train_CD4T_exp.h5ad")
    test_b_path = os.path.join(data_dir, "task2_test_B_exp.h5ad")
    test_nk_path = os.path.join(data_dir, "task2_test_NK_exp.h5ad")
    for p in [train_path, test_b_path, test_nk_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError("Required file not found: " + p)

    train = sc.read_h5ad(train_path)
    train.obs["split"] = "train"

    test_b = sc.read_h5ad(test_b_path)
    test_nk = sc.read_h5ad(test_nk_path)
    ctrl_b = test_b[test_b.obs["perturbation_status"] == "Control"].copy()
    ctrl_nk = test_nk[test_nk.obs["perturbation_status"] == "Control"].copy()
    ctrl_b.obs["split"] = "test_control"
    ctrl_nk.obs["split"] = "test_control"

    combined = sc.concat([train, ctrl_b, ctrl_nk], join="outer", index_unique=None)
    combined.obs_names_make_unique()
    combined.write_h5ad(out_path)
    n_train, n_b, n_nk = train.shape[0], ctrl_b.shape[0], ctrl_nk.shape[0]
    print(f"Written {combined.shape[0]} cells (train {n_train} + test_B control {n_b} + test_NK control {n_nk}) => {out_path}")

if __name__ == "__main__":
    main()
