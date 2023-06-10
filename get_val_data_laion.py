import os
import glob
import shutil
import numpy as np
from annotator.canny import CannyDetector
from annotator.util import HWC3
import cv2
import json
from PIL import Image


ROOT = "/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/"


# get names of all files in directory
list_of_files = glob.glob(os.path.join(ROOT, 'data/laion-art/images/*'))
list_of_prompts = glob.glob(os.path.join(ROOT, 'data/laion-art/prompts/*'))
r_indx = np.random.randint(0, len(list_of_files), 64)
list_of_files = [list_of_files[i] for i in r_indx]
list_of_prompts = [list_of_prompts[i] for i in r_indx]

with open(os.path.join(ROOT, 'data/laion-art/val_data_test.json'), 'wt') as f:
    for i in range(len(list_of_files)):
        with open(list_of_prompts[i], 'r') as p:
            prompt = p.readline().strip()
        f.write(json.dumps({'target': list_of_files[i], 'source': list_of_files[i].replace('images', 'edges'), 'prompt': prompt, 'ds_label': 'laion-art'}) + '\n')


os.makedirs(os.path.join(ROOT, 'data/laion-art/edges/'), exist_ok=True)
apply_canny = CannyDetector()

for i in range(len(list_of_files)):
    target = cv2.imread(list_of_files[i])
    detected_map = apply_canny(target, 100, 200)
    Image.fromarray(detected_map).save(list_of_files[i].replace('images', 'edges'))
    # detected_map = HWC3(detected_map)
