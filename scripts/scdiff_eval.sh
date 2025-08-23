trap "echo ERROR && exit 1" ERR

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}

NAME=v7.5
OFFLINE_SETTINGS="--wandb_offline t"
# --------------------

HOMEDIR=$(dirname $(dirname $(realpath $0)))
cd $HOMEDIR
echo HOMEDIR=$HOMEDIR  # PertBench

dataset_name=fig1_task1_B

data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=task1_train_B_exp.h5ad"
data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=task1_valid_B_exp.h5ad"

python src/scDiff/main.py \
    --base configs/scdiff/eval_perturbation.yaml \
    --name ${NAME} \
    --logdir ${LOGDIR} \
    --postfix perturbation_${NAME} \
    ${OFFLINE_SETTINGS} \
    ${data_settings}

