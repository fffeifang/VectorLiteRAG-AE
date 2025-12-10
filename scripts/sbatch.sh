#!/bin/bash

#SBATCH --job-name=trainer
#SBATCH --account=gts-dmahajan7
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=192G
#SBATCH --time=08:00:00
#SBATCH -qembers
#SBATCH -o log/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jkim4112@gatech.edu

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
source activate ./scripts/vlite

srun python -m index.trainer -d orcas2k -o build_fs -n 1024 -g