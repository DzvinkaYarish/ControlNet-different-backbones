import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from torchmetrics.functional.multimodal import clip_score
from annotator.util import HWC3, resize_image
from aesthetics_predictor import AestheticsPredictor
from fid import fid_score
from functools import partial
import cv2
import time


def apply_canny(img):
    return cv2.Canny(img, 100, 200)

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    print(prompts)
    clip_score = clip_score_fn(images, prompts).detach()
    clip_score = np.array([np.round((clip_score.numpy()), 4)])
    return clip_score

def rmse(x, y):
    return np.mean(np.sqrt(np.mean((x - y) ** 2, axis=1)))

class ImageLogger(Callback):
    def __init__(self, save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, val_dataloader=None):
        super().__init__()
        self.save_dir = save_dir
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
        self.aesthetics_predictor = AestheticsPredictor()
        self.val_dataloader = val_dataloader

        if self.val_dataloader:
            self.get_target_clip_scores()

    def get_target_clip_scores(self):
        self.target_clip_scores =  {'things': [], 'laion-art':[], 'CC3M':[]}
        for batch in self.val_dataloader:
            target = torch.Tensor((batch['jpg'] + 1.0) * 127.5).to(torch.uint8)
            self.target_clip_scores[batch['ds_label'][0]].append(calculate_clip_score(target, batch['txt']))

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.save_dir, "image_log", split)
        ds_label = images['ds_label'][0]
        del images['ds_label']
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images_to_log = images[k][:N]
            grid = torchvision.utils.make_grid(images_to_log, nrow=4)
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
            sampling_times = []
            with torch.no_grad():
                if self.val_dataloader:
                    for dl_batch in self.val_dataloader:
                        s = time.time()
                        imgs = pl_module.log_images(dl_batch, split=split, **self.log_images_kwargs)
                        sampling_times.append(time.time() - s)
                        imgs['ds_label'] = dl_batch['ds_label']
                        # imgs['target'] = dl_batch['jpg']
                        all_images.append(imgs)
                else:
                    s = time.time()
                    imgs = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                    sampling_times.append(time.time() - s)
                    imgs['ds_label'] = ''
                    # imgs['target'] = batch['jpg']
                    all_images.append(imgs)

            for i, images in enumerate(all_images):
                for k in images:
                    if k != 'ds_label':
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)
                            if self.rescale:
                                images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                            images[k] = (images[k] * 255).to(torch.uint8)

                self.log_local(split, images,
                               pl_module.global_step, pl_module.current_epoch, i)

            # log metrics
            metrics = {"clip_score": {'things': [], 'laion-art': [], 'CC3M':[]}, 'edge_rmse': [], 'delta_clip_score': {'things': [], 'laion-art':[], 'CC3M':[]}, 'aesthetics_score': [], 'sampling_time': np.round(np.mean(sampling_times), 2)}
            if self.val_dataloader:
                for i, dl_batch in enumerate(self.val_dataloader):
                    # all_images[i]['samples_cfg_scale_9.00'] returns torch tensor of 
                    # shape batch, 3, width, height with pixel range [0, 255] (uint8)
                    # CLIP expects RGB batch of shape batch, 3, 256, 256 with pixel range [0, 1] normalized with 
                    # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                    # https://github.com/openai/CLIP/blob/main/clip/clip.py
                    metrics['aesthetics_score'].append(np.mean(self.aesthetics_predictor.inference(all_images[i]['samples_cfg_scale_9.00'].float() / 255.0).cpu().detach().numpy()))
                    metrics['clip_score'][dl_batch['ds_label'][0]].append(calculate_clip_score(all_images[i]['samples_cfg_scale_9.00'], dl_batch['txt']))
                    # TODO canny is hardcoded here but we might use other conditions in the future
                    generated_edges = np.array([HWC3(apply_canny(img.permute(1,2,0).numpy())) for img in all_images[i]['samples_cfg_scale_9.00']])
                    metrics['edge_rmse'].append(rmse(generated_edges, dl_batch['hint'].numpy()))

                for ds_l in ['things', 'laion-art','CC3M']:
                    metrics['delta_clip_score'][ds_l] = np.mean(np.concatenate(metrics['clip_score'][ds_l]) - np.concatenate(self.target_clip_scores[ds_l]))
                    metrics[f'delta_clip_score_{ds_l}'] = metrics['delta_clip_score'][ds_l]
                    del metrics['delta_clip_score'][ds_l]

                    metrics['clip_score'][ds_l] = np.mean(metrics['clip_score'][ds_l])
                    metrics[f'clip_score_{ds_l}'] = metrics['clip_score'][ds_l]
                    del metrics['clip_score'][ds_l]


                metrics['edge_rmse'] = np.mean(metrics['edge_rmse'])
                metrics['aesthetics_score'] = np.mean(metrics['aesthetics_score'])
                try:  # sometimes the calculation fails with "sqrtm: array must not contain infs or NaNs"
                    metrics['fid_score'] = fid_score.calculate_fid(all_images, self.val_dataloader, batch_size=32, 
                        device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048, num_workers=4)
                except:
                    pass
                pl_module.logger.log_metrics(metrics, step=pl_module.global_step)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_train_end(self, trainer, pl_module):
        print('ON TRAIN END')
        self.log_img(pl_module, None, 0, split="train")

    def on_exception(self, trainer, pl_module, exception):
        print('ON EXCEPTION')
        self.log_img(pl_module, None, 0, split="train")
