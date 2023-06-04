from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, ValDataset
from laion_dataset import LAIONDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os
from datetime import datetime


# Configs
ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
#ROOT = "./"
experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_sd15"
config_path = os.path.join(ROOT, './models/cldm_v15.yaml')
resume_path = os.path.join(ROOT, 'models/control_sd15_SD_ini.ckpt')
exp_path = os.path.join(ROOT, f'experiments/debug/{experiment_name}')
# exp_path = os.path.join(ROOT, './experiments/laion_SD/')
batch_size = 4
logger_freq = 500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
# dataset = LAIONDataset()
dataloader = DataLoader(MyDataset(), num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(ValDataset(), num_workers=0, batch_size=batch_size, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq, val_dataloader=val_dataloader, log_images_kwargs={"ddim_steps":50})
wandb_logger = WandbLogger(save_dir=exp_path, name=experiment_name, project="ControlNet")
trainer = pl.Trainer(accelerator='gpu', gpus=1, precision=32, callbacks=[logger], 
    logger=wandb_logger, default_root_dir=exp_path, max_steps=5000)


# Train!
trainer.fit(model, dataloader)
