import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
import tqdm

from os.path import join
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticsPredictor:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/models/aesthetics-predictor/sac+logos+ava1-l14-linearMSE.pth")
        model.load_state_dict(s)

        model.to(device)
        model.eval()

        clip_model, _ = clip.load("ViT-L/14", device=device)

        self.device = device
        self.model = model
        self.clip_model = clip_model
        self.preprocess = preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def inference(self, batch):
        """
        Batch is expected to be of shape batch, 3, width, height with pixel range [0, 1]
        """
        images = self.preprocess(batch).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))

        return prediction

if __name__ == "__main__":
    model = AestheticsPredictor()
    print(np.mean(model.inference(torch.rand((4, 3, 256, 256))).cpu().detach().numpy()))
