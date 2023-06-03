#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=4000
####### --nodelist=falcon5

ROOT="/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones"

module load any/python/3.8.3-conda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate controlnet

nvidia-smi

gcc --version

# python tool_add_control.py $ROOT/models/v1-5-pruned.ckpt $ROOT/models/control_sd15_SD_ini.ckpt
python tutorial_train.py
