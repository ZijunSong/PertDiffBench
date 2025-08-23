# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/dfcontrol_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/dfcontrol_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/dfcontrol.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed123_test_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed123_test_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed123_test.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed123_train_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed123_train_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed123_train.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed345_test_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed345_test_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed345_test.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed345_train_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed345_train_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed345_train.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed567_test_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed567_test_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed567_test.h5ad'

# python scripts/tools/fig2_task1_merge.py \
#     --exp 'data/fig2/task1_unseen_pert/seed567_train_exp.csv' \
#     --meta 'data/fig2/task1_unseen_pert/seed567_train_meta.csv' \
#     --output 'data/fig2/task1_unseen_pert/seed567_train.h5ad'

python scripts/tools/fig2_task1_split.py \
    --control data/fig2/task1_unseen_pert/dfcontrol.h5ad \
    --train data/fig2/task1_unseen_pert/seed123_train.h5ad \
    --test data/fig2/task1_unseen_pert/seed123_test.h5ad \
    --output_train data/fig2/task1_unseen_pert/seed123_control_train.h5ad \
    --output_test data/fig2/task1_unseen_pert/seed123_control_test.h5ad 

python scripts/tools/fig2_task1_split.py \
    --control data/fig2/task1_unseen_pert/dfcontrol.h5ad \
    --train data/fig2/task1_unseen_pert/seed345_train.h5ad \
    --test data/fig2/task1_unseen_pert/seed345_test.h5ad \
    --output_train data/fig2/task1_unseen_pert/seed345_control_train.h5ad \
    --output_test data/fig2/task1_unseen_pert/seed345_control_test.h5ad 

python scripts/tools/fig2_task1_split.py \
    --control data/fig2/task1_unseen_pert/dfcontrol.h5ad \
    --train data/fig2/task1_unseen_pert/seed567_train.h5ad \
    --test data/fig2/task1_unseen_pert/seed567_test.h5ad \
    --output_train data/fig2/task1_unseen_pert/seed567_control_train.h5ad \
    --output_test data/fig2/task1_unseen_pert/seed567_control_test.h5ad 
