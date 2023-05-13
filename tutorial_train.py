from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os


# Configs
ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
# ROOT = "./"
resume_path = os.path.join(ROOT, 'models/control_lite_conv_ini.ckpt')
exp_path = os.path.join(ROOT, 'experiments/fillin50K_lite_conv_lr_1e-4/')
batch_size = 8
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_lite.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator='gpu', gpus=1, precision=32, callbacks=[logger], default_root_dir=exp_path)


# Train!
trainer.fit(model, dataloader)
