import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from torchmetrics.functional.multimodal import clip_score
from annotator.util import HWC3, resize_image
from functools import partial
import cv2


def apply_canny(img):
    return cv2.Canny(img, 100, 200)

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(images, prompts).detach()
    return np.array([np.round((clip_score.numpy()), 4)])

def rmse(x, y):
    return np.mean(np.sqrt(np.mean((x - y) ** 2, axis=1)))

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, val_dataloader=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.val_dataloader = val_dataloader

        if self.val_dataloader:
            self.get_target_clip_scores()

    def get_target_clip_scores(self):
        self.target_clip_scores = []
        for batch in self.val_dataloader:
            target = torch.Tensor((batch['jpg'] + 1.0) * 127.5).to(torch.uint8)
            self.target_clip_scores.append(calculate_clip_score(target, batch['txt']))

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        ds_label = images['ds_label'][0]
        del images['ds_label']
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = grid.astype(np.uint8)
            filename = "{}_{}_gs-{:06}_e-{:06}_b-{:06}.png".format(ds_label, k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            all_images = []
            with torch.no_grad():
                if self.val_dataloader:
                    for dl_batch in self.val_dataloader:
                        imgs = pl_module.log_images(dl_batch, split=split, **self.log_images_kwargs)
                        imgs['ds_label'] = dl_batch['ds_label']
                        # imgs['target'] = dl_batch['jpg']
                        all_images.append(imgs)
                else:
                    imgs = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                    imgs['ds_label'] = ''
                    # imgs['target'] = batch['jpg']
                    all_images.append(imgs)

            for i, images in enumerate(all_images):
                for k in images:
                    if k != 'ds_label':
                        N = min(images[k].shape[0], self.max_images)
                        images[k] = images[k][:N]
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)
                            images[k] = (images[k] * 255).to(torch.uint8)

                self.log_local(pl_module.logger.save_dir, split, images,
                               pl_module.global_step, pl_module.current_epoch, i)

            # log metrics
            metrics = {"clip_score": [], 'edge_rmse': [], 'delta_clip_score': []}
            if self.val_dataloader:
                for i, dl_batch in enumerate(self.val_dataloader):
                    metrics['clip_score'].append(calculate_clip_score(all_images[i]['samples_cfg_scale_9.00'], dl_batch['txt']))
                    generated_edges = np.array([HWC3(apply_canny(img.permute(1,2,0).numpy())) for img in all_images[i]['samples_cfg_scale_9.00']])
                    metrics['edge_rmse'].append(rmse(generated_edges, dl_batch['hint'].numpy()))

                metrics['delta_clip_score'] = np.mean(np.concatenate(metrics['clip_score']) - np.concatenate(self.target_clip_scores))
                metrics['clip_score'] = np.mean(metrics['clip_score'])
                metrics['edge_rmse'] = np.mean(metrics['edge_rmse'])
                pl_module.logger.log_metrics(metrics, step=pl_module.global_step)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
