#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
# export ASCEND_GLOBAL_LOG_LEVEL=2
# export SLOG_PRINT_TO_STDOUT=2
# export MS_ENABLE_FORMAT_MODE=1
# export MINDSPORE_DUMP_CONFIG='/share-nfs/w50035851/code/msver/dump.json'
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
dir=device_$1_$4
workspace=$(pwd)
export HCCL_DETERMINISTIC=1
rm -rf $dir
mkdir $dir
cp ./*.py $dir
rm -rf log/fin*.txt
cd $dir
echo "start training"
ttl=8
port=8448
export MS_WORKER_NUM=$ttl # number of worker processes in the cluster (8)
export MS_SCHED_HOST=127.0.0.1 # scheduler host (loopback)
export MS_SCHED_PORT=$port # scheduler port
# launch 8 worker training processes
export MS_ROLE=MS_SCHED # run this process as MS_SCHED
python ./$1.py --batch $2 --epoch $3 --dist --data $4 --load_pretrain --workpath  > scheduler.log 2>&1 &
for((i=1;i<$ttl;i++));
do
 export MS_ROLE=MS_WORKER # run this process as MS_WORKER
 export MS_NODE_ID=$i # process id (optional)
    python ./$1.py --batch $2 --epoch $3  --dist --data $4 --load_pretrain > worker_$i.log 2>&1 &
    
done
export MS_ROLE=MS_WORKER # run this process as MS_WORKER
export MS_NODE_ID=0 # process id (optional)
python ./$1.py --batch $2 --epoch $3 --dist --data $4 --load_pretrain