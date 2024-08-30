#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 48:00:00 -o out -e err
#SBATCH -p gpu_4090
#SBATCH --job-name=test

# source /share/home/bjiangch/group-zyl/.bash_profile
# conda environment
#source /share/apps/Anaconda/anaconda3/etc/profile.d/conda.sh
module load anaconda/2020.11
source activate py39
python test.py
python plot_data.py
