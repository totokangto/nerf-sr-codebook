#!/usr/bin/bash

#SBATCH -J nerf-sr
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y5
#SBATCH -t 2-0
#SBATCH -o logs/slurm-%A.out

bash scripts/train_llff_cb.sh
# python warp.py
exit 0
