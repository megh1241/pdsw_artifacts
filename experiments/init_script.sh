#!/bin/bash
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# note: disable registration cache for verbs provider for now; see
#       discussion in https://github.com/ofiwg/libfabric/issues/5244
export FI_MR_CACHE_MAX_COUNT=0
# use shared recv context in RXM; should improve scalability
export FI_OFI_RXM_USE_SRX=1
export LD_LIBRARY_PATH=$HOME/install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/experiments/cpp-store:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/miniconda/lib:$LD_LIBRARY_PATH
soft add +mvapich2
soft add +cuda-11.0.2
source $HOME/miniconda/bin/activate
set +eu
conda activate /home/mmadhya1/dh-cooley-clean
set -eu
