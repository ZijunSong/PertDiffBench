# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/mouse_control_meta.csv \
#     data/fig2/task3_cross_species/mouse_control_exp.csv \
#     data/fig2/task3_cross_species/mouse_control.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/mouse_LPS6_exp.csv \
#     data/fig2/task3_cross_species/mouse_LPS6_exp.csv \
#     data/fig2/task3_cross_species/mouse_ifn.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/pig_control_meta.csv \
#     data/fig2/task3_cross_species/pig_control_exp.csv \
#     data/fig2/task3_cross_species/pig_control.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/pig_LPS6_meta.csv \
#     data/fig2/task3_cross_species/pig_LPS6_exp.csv \
#     data/fig2/task3_cross_species/pig_ifn.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/rabbit_control_meta.csv \
#     data/fig2/task3_cross_species/rabbit_control_exp.csv \
#     data/fig2/task3_cross_species/rabbit_control.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/rabbit_LPS6_meta.csv \
#     data/fig2/task3_cross_species/rabbit_LPS6_exp.csv \
#     data/fig2/task3_cross_species/rabbit_ifn.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/rat_control_meta.csv \
#     data/fig2/task3_cross_species/rat_control_exp.csv \
#     data/fig2/task3_cross_species/rat_control.h5ad

# python scripts/tools/fig2_task3_merge_csv.py \
#     data/fig2/task3_cross_species/rat_LPS6_meta.csv \
#     data/fig2/task3_cross_species/rat_LPS6_exp.csv \
#     data/fig2/task3_cross_species/rat_ifn.h5ad



python scripts/tools/fig2_task3_merge_h5ad.py \
    data/fig2/task3_cross_species/mouse_control.h5ad \
    data/fig2/task3_cross_species/mouse_ifn.h5ad \
    data/fig2/task3_cross_species/mouse_control_ifn.h5ad

python scripts/tools/fig2_task3_merge_h5ad.py \
    data/fig2/task3_cross_species/pig_control.h5ad \
    data/fig2/task3_cross_species/pig_ifn.h5ad \
    data/fig2/task3_cross_species/pig_control_ifn.h5ad

python scripts/tools/fig2_task3_merge_h5ad.py \
    data/fig2/task3_cross_species/rabbit_control.h5ad \
    data/fig2/task3_cross_species/rabbit_ifn.h5ad \
    data/fig2/task3_cross_species/rabbit_control_ifn.h5ad

python scripts/tools/fig2_task3_merge_h5ad.py \
    data/fig2/task3_cross_species/rat_control.h5ad \
    data/fig2/task3_cross_species/rat_ifn.h5ad \
    data/fig2/task3_cross_species/rat_control_ifn.h5ad