python scripts/baseline/train_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_train_random1_bulkRNAseq_exp.h5ad \
    --save-weight-dir checkpoints/fig1/task2/random1/mlp_ddpm_mlp \
    --sampled-dir samples/fig1/task2/random1/mlp_ddpm_mlp \
    --gene-nums 2969

python scripts/baseline/train_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_train_random2_bulkRNAseq_exp.h5ad \
    --save-weight-dir checkpoints/fig1/task2/random2/mlp_ddpm_mlp \
    --sampled-dir samples/fig1/task2/random2/mlp_ddpm_mlp \
    --gene-nums 2969

python scripts/baseline/train_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_train_random3_bulkRNAseq_exp.h5ad \
    --save-weight-dir checkpoints/fig1/task2/random3/mlp_ddpm_mlp \
    --sampled-dir samples/fig1/task2/random3/mlp_ddpm_mlp \
    --gene-nums 2969

python scripts/baseline/eval_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_test_random1_bulkRNAseq_exp.h5ad \
    --ckpt checkpoints/fig1/task2/random1/mlp_ddpm_mlp/scrna_ddpm_epoch1000.pt \
    --out_h5ad samples/fig1/task2/random1/mlp_ddpm_mlp/synthetic_ifn.h5ad \
    --gene-nums 2969 \
    --n_samples 100

python scripts/baseline/eval_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_test_random2_bulkRNAseq_exp.h5ad \
    --ckpt checkpoints/fig1/task2/random2/mlp_ddpm_mlp/scrna_ddpm_epoch1000.pt \
    --out_h5ad samples/fig1/task2/random2/mlp_ddpm_mlp/synthetic_ifn.h5ad \
    --gene-nums 2969 \
    --n_samples 100

python scripts/baseline/eval_mlp_ddpm_mlp.py \
    --config configs/baselines/mlp_ddpm_mlp.yaml \
    --data-path data/fig1/task2/task2_test_random3_bulkRNAseq_exp.h5ad \
    --ckpt checkpoints/fig1/task2/random3/mlp_ddpm_mlp/scrna_ddpm_epoch1000.pt \
    --out_h5ad samples/fig1/task2/random3/mlp_ddpm_mlp/synthetic_ifn.h5ad \
    --gene-nums 2969 \
    --n_samples 100
