#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=4000
####### --nodelist=falcon5

module load any/python/3.8.3-conda

conda activate controlnet

nvidia-smi

gcc --version

# P_PATH=/gpfs/space/home/dzvenymy/.conda/envs/controlnet/bin/python
P_PATH=/gpfs/space/home/zaliznyi/miniconda3/envs/controlnet/bin/python


#/gpfs/space/home/dzvenymy/.conda/envs/controlnet/bin/python tool_add_control.py /gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/models/v1-5-pruned.ckpt /gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/models/control_lite_ini.ckpt

# $P_PATH tutorial_train.py --max_steps 15000 --experiment_name laion_sd_fixed_steps --logger_freq 500 --dataset laion --resume_path control_sd15_SD_ini.ckpt --model_config cldm_v15.yaml --learning_rate 1e-5
#/gpfs/space/home/dzvenymy/.conda/envs/controlnet/bin/python tutorial_train.py --max_time 00:2:00:00 --experiment_name fillin50k_mlp_fixed_time --logger_freq 500

# $P_PATH tutorial_train.py --dataset laion --max_time 00:3:00:00 --experiment_name laion_sd_fixed_time_1 --resume_path control_sd15_SD_ini.ckpt --model_config cldm_v15.yaml --learning_rate 1e-5 --logger_dir /gpfs/space/home/zaliznyi/wandb
# $P_PATH tutorial_train.py --dataset laion --max_time 00:3:00:00 --experiment_name laion_mlp_fixed_time_1 --resume_path control_lite_ini.ckpt --model_config cldm_lite_mlp.yaml --learning_rate 1e-4 --logger_dir /gpfs/space/home/zaliznyi/wandb
$P_PATH tutorial_train.py --dataset laion --max_time 00:3:00:00 --experiment_name laion_conv_fixed_time_1 --resume_path control_lite_conv_ini.ckpt --model_config cldm_lite_conv.yaml --learning_rate 1e-4 --logger_dir /gpfs/space/home/zaliznyi/wandb
