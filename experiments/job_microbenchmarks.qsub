#!/bin/bash
#COBALT -A VeloC
#COBALT -n 2
#COBALT -t 0:15:00
#COBALT -q debug
#COBALT --mode script

# note: disable registration cache for verbs provider for now; see
#       discussion in https://github.com/ofiwg/libfabric/issues/5244
export FI_MR_CACHE_MAX_COUNT=0
# use shared recv context in RXM; should improve scalability
export FI_OFI_RXM_USE_SRX=1
export LD_LIBRARY_PATH=$HOME/install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/lib:$LD_LIBRARY_PATH



EXP_DIR=$HOME/experiments
CPP_STORE_DIR=$EXP_DIR/cpp-store
rm $CPP_STORE_DIR/server_str.txt

mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE
head_node=${nodes_array[0]}

cd $CPP_STORE_DIR
mpirun --hosts $head_node -n 1 ./server --thallium_connection_string "ofi+verbs" --provider_id 0 --num_threads 1 --storage_backend "rocksdb" &

sleep 1

INIT_SCRIPT=$EXP_DIR/init_script.sh
source $INIT_SCRIPT
cd $HOME
client_node=${nodes_array[1]}
mpiexec -n 1 --hosts $client_node $(which python) -m mpi4py ./experiments/microbenchmark.py --save_directory "/home/mmadhya1/rdma_res4/"
