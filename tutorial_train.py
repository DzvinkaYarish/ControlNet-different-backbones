from share import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tutorial_dataset import Fill50kDataset, ValDataset
from laion_dataset import LAIONDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import argparse
import os
from datetime import datetime

ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
#ROOT = "./"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='laion')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_time', type=str, default="00:3:00:00")
    parser.add_argument('--experiment_name', type=str, default='laion_sd_fixed_time')
    parser.add_argument('--resume_path', type=str, default='control_sd15_SD_ini.ckpt') # for sd: control_sd15_SD_ini.ckpt,  for mlp: control_lite_ini.ckpt, for conv: control_lite_conv_ini.ckpt
    parser.add_argument('--model_config', type=str, default='cldm_v15.yaml') # for sd: cldm_v15.yaml, for mlp: cldm_lite_mlp.yaml, for conv: cldm_lite_conv.yaml
    parser.add_argument('--sd_locked', type=bool, default=True)
    parser.add_argument('--only_mid_control', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # 1e-5 for SD, 1e-4 for lite
    parser.add_argument('--logger_freq', type=int, default=500)
    parser.add_argument('--logger_dir', type=str, default='/gpfs/space/home/zaliznyi/wandb')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = vars(args)
    config['resume_path'] = os.path.join(ROOT, 'models', config['resume_path'])
    config['model_config'] = os.path.join(ROOT, 'models', config['model_config'])
    exp_path = os.path.join(ROOT, 'experiments', config['experiment_name'])

    print(config)

    # experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_sd15"

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config['model_config']).cpu()
    model.load_state_dict(load_state_dict(config['resume_path'], location='cpu'))
    model.learning_rate = config['learning_rate']
    model.sd_locked = config['sd_locked']
    model.only_mid_control = config['only_mid_control']


    # Misc
    if config['dataset'] == 'fill50k':
        dataset = Fill50kDataset()
    else:
        dataset = LAIONDataset()

    dataloader = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(ValDataset(config['dataset']), num_workers=0, batch_size=config['batch_size'], shuffle=False)

    logger = ImageLogger(save_dir=exp_path, batch_frequency=config['logger_freq'], val_dataloader=val_dataloader, log_images_kwargs={"ddim_steps":50, "N":8})
    wandb_logger = WandbLogger(save_dir=exp_path, config=config, name=config['experiment_name'], project="ControlNetTartu", dir=config['logger_dir'])
    trainer = pl.Trainer(accelerator='gpu', gpus=1, precision=32, callbacks=[logger],
        logger=wandb_logger, default_root_dir=exp_path, max_steps=config['max_steps'], max_time=config['max_time'])


    # Train!
    trainer.fit(model, dataloader)

    # Log final images
    # logger.log_img(model, None, 0)