#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=4000
####### --nodelist=falcon5

module load any/python/3.8.3-conda

source ~/miniconda3/etc/profile.d/conda.sh

conda activate controlnet

nvidia-smi

gcc --version

#/gpfs/space/home/dzvenymy/.conda/envs/controlnet/bin/python tool_add_control.py /gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/models/v1-5-pruned.ckpt /gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/models/control_lite_ini.ckpt

# /gpfs/space/home/dzvenymy/.conda/envs/controlnet/bin/python tutorial_train.py

/gpfs/space/home/zaliznyi/miniconda3/envs/controlnet/bin/python tutorial_train.py
