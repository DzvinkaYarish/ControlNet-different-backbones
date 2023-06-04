import json
import cv2
import numpy as np
from annotator.util import resize_image, HWC3

from torch.utils.data import Dataset

import os


ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"
DATA = "./data/laion-art/"


class LAIONDataset(Dataset):
    def __init__(self, canny_low=100, canny_high=200):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.data = []

        # only first 30 directories seem to be correct, others might be corrupted (TODO redownload data)
        directories = [f"000{i}" if i > 9 else f"0000{i}" for i in range(1)]
        for d in directories:
            print(f"Parsing directory {d}...")
            directory_path = os.path.join(ROOT, DATA, d)
            for filename in os.listdir(directory_path):
                name = os.path.splitext(filename)[0]
                jpg = f"{name}.jpg"
                txt = f"{name}.txt"
                jpg_path = os.path.join(directory_path, jpg)
                txt_path = os.path.join(directory_path, txt)
                if os.path.exists(jpg_path) and os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        prompt = f.read()
                    self.data.append({'target': os.path.join(d, jpg), 'prompt': prompt})
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']  # TODO pregenerate source (hint) data
        target_filename = item['target']
        prompt = item['prompt']

        #source = cv2.imread(os.path.join(ROOT, DATA, source_filename))
        target = cv2.imread(os.path.join(ROOT, DATA, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        #source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize source and target
        #source = resize_image(source, 256)
        target = resize_image(target, 256)

        # Create source (hint) image
        source = HWC3(cv2.Canny(target, self.canny_low, self.canny_high))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

