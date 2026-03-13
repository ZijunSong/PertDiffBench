# 原始数据目录与输出目录（请在 PertDiffBench 项目根目录下运行本脚本）
DATA_ORI=/data/ppnm/data/PertDiffBench/data_ori/fig2/task3_cross_species
DATA_OUT=/data/ppnm/data/PertDiffBench/data/fig2_task3_cross_species
mkdir -p "$DATA_OUT"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/mouse_control_meta.csv" \
    "$DATA_ORI/mouse_control_exp.csv" \
    "$DATA_OUT/mouse_control.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/mouse_LPS6_exp.csv" \
    "$DATA_ORI/mouse_LPS6_exp.csv" \
    "$DATA_OUT/mouse_ifn.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/pig_control_meta.csv" \
    "$DATA_ORI/pig_control_exp.csv" \
    "$DATA_OUT/pig_control.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/pig_LPS6_meta.csv" \
    "$DATA_ORI/pig_LPS6_exp.csv" \
    "$DATA_OUT/pig_ifn.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/rabbit_control_meta.csv" \
    "$DATA_ORI/rabbit_control_exp.csv" \
    "$DATA_OUT/rabbit_control.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/rabbit_LPS6_meta.csv" \
    "$DATA_ORI/rabbit_LPS6_exp.csv" \
    "$DATA_OUT/rabbit_ifn.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/rat_control_meta.csv" \
    "$DATA_ORI/rat_control_exp.csv" \
    "$DATA_OUT/rat_control.h5ad"

python scripts/tools/fig2_task3_merge_csv.py \
    "$DATA_ORI/rat_LPS6_meta.csv" \
    "$DATA_ORI/rat_LPS6_exp.csv" \
    "$DATA_OUT/rat_ifn.h5ad"



python scripts/tools/fig2_task3_merge_h5ad.py \
    "$DATA_OUT/mouse_control.h5ad" \
    "$DATA_OUT/mouse_ifn.h5ad" \
    "$DATA_OUT/mouse_control_ifn.h5ad"

python scripts/tools/fig2_task3_merge_h5ad.py \
    "$DATA_OUT/pig_control.h5ad" \
    "$DATA_OUT/pig_ifn.h5ad" \
    "$DATA_OUT/pig_control_ifn.h5ad"

python scripts/tools/fig2_task3_merge_h5ad.py \
    "$DATA_OUT/rabbit_control.h5ad" \
    "$DATA_OUT/rabbit_ifn.h5ad" \
    "$DATA_OUT/rabbit_control_ifn.h5ad"

python scripts/tools/fig2_task3_merge_h5ad.py \
    "$DATA_OUT/rat_control.h5ad" \
    "$DATA_OUT/rat_ifn.h5ad" \
    "$DATA_OUT/rat_control_ifn.h5ad"