# 1. Train the Autoencoder
cd src/scDiffusion/VAE
python VAE_train.py \
    --data_dir ../../../data/fig1/task2/task2_train_random1_bulkRNAseq_exp.h5ad \
    --num_genes 2969 \
    --state_dict ../../../checkpoints/scimilarity/model_v1.1 \
    --save_dir ../../../checkpoints/scdiffusion/vae_checkpoint/VAE_random1

python VAE_train.py \
    --data_dir ../../../data/fig1/task2/task2_train_random2_bulkRNAseq_exp.h5ad \
    --num_genes 2969 \
    --state_dict ../../../checkpoints/scimilarity/model_v1.1 \
    --save_dir ../../../checkpoints/scdiffusion/vae_checkpoint/VAE_random2

python VAE_train.py \
    --data_dir ../../../data/fig1/task2/task2_train_random3_bulkRNAseq_exp.h5ad \
    --num_genes 2969 \
    --state_dict ../../../checkpoints/scimilarity/model_v1.1 \
    --save_dir ../../../checkpoints/scdiffusion/vae_checkpoint/VAE_random3

# 2. Train the diffusion backbone
cd ..
python cell_train.py

# 3. rain the classifier
python classifier_train.py
