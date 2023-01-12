#!/bin/bash

#SBATCH --job-name=experiment       # 作业名
#SBATCH --partition=dgx2       # dgx2 队列
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6     # 1:6 的 GPU:CPU 配比
#SBATCH --gres=gpu:1           # 2 块 GPU
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda
module load miniconda3
source activate pytorch-env
module load pytorch/1.6.0
cd ~/DRG
python train/train.py -c config/drg_baseline.yml