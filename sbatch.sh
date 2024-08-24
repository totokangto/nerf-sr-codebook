#!/usr/bin/bash

#SBATCH -J Nerfsr-Train-llff-downX-refine-patchsamearea8-enhancernetwork
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.outs

cat $0
pwd
which python
hostname

# train: original llff nerf
# bash scripts/train_llff.sh

# train
# bash scripts/train_llff_downX.sh

# test
# bash scripts/test_llff_downX.sh

# # Refinement
# python warp.py
bash scripts/train_llff_refine.sh

# test
# bash scripts/test_llff_refine.sh

# bash scripts/train_blender.sh
# bash scripts/train_blender_downX.sh

# bash scripts/test_blender.sh
# bash scripts/test_blender_downX.sh

exit 0
