#!/bin/bash
#SBATCH --gpus=1
#SBATCH  -o out -e err
#SBATCH -p gpu_4090
#SBATCH --job-name=ECL

# source /share/home/bjiangch/group-zyl/.bash_profile
# conda environment
#source /share/apps/Anaconda/anaconda3/etc/profile.d/conda.sh
#conda_env=dmff_0.2.0
module load anaconda/2020.11
source activate py39
export OMP_NUM_THREADS=16
#path to save the code
path="/HOME/scw6851/run/junmin/EANN_MOL/program/"

#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
#WORLD_SIZE=`expr $PBS_NUM_NODES \* $NPROC_PER_NODE`

#MASTER=`/bin/hostname -s`

#MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
#        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`

#You will want to replace this
COMMAND="$path "
#conda activate $conda_env
# cd $PBS_O_WORKDIR 
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$PBS_NUM_NODES --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$MPORT $COMMAND > out
python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=1 --standalone $COMMAND > out
# python3 -m torch.distributed.run --nproc_per_node=1 --max_restarts=0 --nnodes=1 --standalone /share/home/junmin/group/DMFF/examples/eann/REANN/reann
