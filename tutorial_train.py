from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, ValDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os


# Configs
# ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
ROOT = "./"
resume_path = os.path.join(ROOT, 'models/control_sd15_SD_ini.ckpt')
exp_path = os.path.join(ROOT, 'experiments/debug/')
batch_size = 4
logger_freq = 500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataloader = DataLoader(MyDataset(), num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(ValDataset(), num_workers=0, batch_size=batch_size, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq, val_dataloader=val_dataloader, log_images_kwargs={"ddim_steps":50})
trainer = pl.Trainer(accelerator='cpu', gpus=0, precision=32, callbacks=[logger], default_root_dir=exp_path, max_steps=5000)


# Train!
trainer.fit(model, dataloader)
