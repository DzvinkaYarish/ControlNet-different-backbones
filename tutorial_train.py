from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tutorial_dataset import Fill50kDataset, ValDataset
from laion_dataset import LAIONDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os
from datetime import datetime

ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
#ROOT = "./"

config = {
"experiment_name": 'fillin50k_mlp_fixed_time',
"resume_path": os.path.join(ROOT, 'models/control_lite_conv_ini.ckpt'), # for sd: control_sd15_SD_ini.ckpt,  for mlp and conv: control_lite_conv_ini.ckpt
"model_config": os.path.join(ROOT, 'models/cldm_lite_mlp.yaml'), #for sd: cldm_v15.yaml, for mlp: cldm_lite_mlp.yaml, for conv: cldm_lite_conv.yaml
"sd_locked": True,
"only_mid_control": False,
"learning_rate": 1e-4,
"batch_size": 8,
"logger_freq": 500,
"dataset": "fill50k",
"max_steps": None,
"max_time": "00:2:00:00", # time format: "00:2:00:00" or None if max_steps is set
}


# Configs

# experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_sd15"

exp_path = os.path.join(ROOT, 'experiments', config['experiment_name'])

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config['model_config']).cpu()
model.load_state_dict(load_state_dict(config['resume_path'], location='cpu'))
model.learning_rate = config['learning_rate']
model.sd_locked =config['sd_locked']
model.only_mid_control = config['only_mid_control']


# Misc
if config['dataset'] == 'fill50k':
    dataset = Fill50kDataset()
else:
    dataset = LAIONDataset()

dataloader = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(ValDataset(config['dataset']), num_workers=0, batch_size=config['batch_size'], shuffle=False)

logger = ImageLogger(save_dir=exp_path, batch_frequency=config['logger_freq'], val_dataloader=val_dataloader, log_images_kwargs={"ddim_steps":50, "N":8})
wandb_logger = WandbLogger(save_dir=exp_path, config=config, name=config['experiment_name'], project="ControlNet", dir='/gpfs/space/home/dzvenymy/wandb')
trainer = pl.Trainer(accelerator='gpu', gpus=1, precision=32, callbacks=[logger], 
    logger=wandb_logger, default_root_dir=exp_path, max_steps=config['max_steps'], max_time=config['max_time'])


# Train!
trainer.fit(model, dataloader)
