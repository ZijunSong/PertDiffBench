#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
# export ASCEND_GLOBAL_LOG_LEVEL=2
# export SLOG_PRINT_TO_STDOUT=2
export HCCL_DETERMINISTIC=1
export MS_ENABLE_FORMAT_MODE=1
export MS_HCCL_CM_INIT=1
# export MINDSPORE_DUMP_CONFIG='/share-nfs/w50035851/code/msver/dump.json'
data='cancer'
start=$3
dir=device$((start/8+1))
rm -rf $dir
mkdir $dir
cp ./*.py ./$dir
cd ./$dir
rm -rf rank*
rm *.log
date
echo "start training"
ttl=32
num=8
ip=$2
batch=4
port=8448
# launch 8 worker training processes
export MS_WORKER_NUM=$ttl # number of worker processes in the cluster (8)
export MS_SCHED_HOST=61.47.2.$ip # scheduler host IP
export MS_SCHED_PORT=$port # scheduler port
export MS_ROLE=MS_WORKER # run this process as MS_WORKER
for((i=$((start+1));i<$((start+num));i++));
do
 export MS_NODE_ID=$i # process id (optional)
    python ./1B_train.py --dist --data $1 --batch $batch --data $data > worker_$i.log 2>&1 &
done
export MS_NODE_ID=$start # process id (optional)
python ./1B_train.py --dist --data $1 --batch $batch --data $data 