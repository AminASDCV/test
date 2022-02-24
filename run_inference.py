import rasterio as rio
from rasterio.windows import Window
from itertools import product
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import TestDrivewayDataset

import glob
import pandas as pd

config = {
    "dir_files": "/mnt/mount-point-directory/datasets2/",
    "batch_size": 12,
    "num_workers": 12,
    "experiment": "exp_1",
    "city": "birmingham-fall-2021",
    "model_trained_multi_gpu": True,
    "reshape_size": 512
}

# Initilization
DIR_FILES = config['dir_files']
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']
EXPERIMENT = config['experiment']
CITY = config['city']
MODEL_TRAINED_ON_MULTIPLE_GPU = config['model_trained_multi_gpu']
RESHAPE_SIZE = config['reshape_size']

path_imgs = glob.glob(os.path.join(DIR_FILES, CITY, 'VRT', 'tiles', '*.jp2'))
path_road_masks = glob.glob(os.path.join(DIR_FILES, CITY, 'mask_road', 'tiles', '*.jp2'))
path_building_fps = glob.glob(os.path.join(DIR_FILES, CITY, 'mask_building_fp', 'tiles', '*.jp2'))
path_to_model = f"/mnt/mount-point-directory/geomate_driveway_detection/experiments/{EXPERIMENT}/model_best.pth.tar"
path_to_save_mask = f"/mnt/mount-point-directory/results/driveways/{CITY}/{EXPERIMENT}"

# sorting to match files
path_imgs.sort()
path_road_masks.sort()
path_building_fps.sort()

df = pd.DataFrame(columns = ['city', 'path_img', 'path_road_mask', 'path_building_fp'])


def add_entry(df_, city_name, path_imgs, path_road_masks, path_building_fps):
    """
    Add a list of img paths and sw_mask paths to the dataframe.
    In future you may change it to generalize using zip(input_values) to form a new row rather than hardcoding.
    """
    for path_img, path_road_mask, path_building_fp in zip(path_imgs, 
                                                          path_road_masks, 
                                                          path_building_fps):
        new_row = {'city': city_name,
                   'path_img': path_img,
                   'path_road_mask': path_road_mask,
                   'path_building_fp': path_building_fp
                  }
        df_ = df_.append(new_row, ignore_index=True)
    return df_


df = add_entry(df, CITY, path_imgs, path_road_masks, path_building_fps)

# Loading the model
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if not torch.cuda.is_available():
    checkpoint = torch.load(path_to_model, map_location=device)
else:
    checkpoint = torch.load(path_to_model)

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1, aux_loss=None)
model.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# original saved file with DataParallel
# create new OrderedDict that does not contain `module.`

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# load params
if MODEL_TRAINED_ON_MULTIPLE_GPU:
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint['state_dict'])


if not os.path.isdir(path_to_save_mask):
    os.makedirs(path_to_save_mask)


test_ds = TestDrivewayDataset(df, config)
test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = True)


model.eval()
model.to(device)

# get the meta data of just one file
src_vrt = rio.open(df['path_img'][0])

for i, sample in tqdm(enumerate(test_loader)):
    
    with torch.no_grad():
        inputRGB = sample[0].to(device)
        outputs = model(inputRGB)
    
    for i in range(len(inputRGB)):

    #     develop output image mask
        pred = (torch.sigmoid(outputs['out'][i].squeeze().cpu()) > 0.5).numpy().astype(int)

        _pred = np.stack([pred, pred, pred], axis = 0) * 255
        _pred = _pred.astype(np.uint8)

        
    #     develop meta file
        meta = {}

        meta['driver'] = 'GTiff'    # rasterio can write only with GTiff driver
        meta['dtype'] = src_vrt.meta['dtype']
        meta['nodata'] = src_vrt.meta['nodata']
        meta['width'] = int(sample[1]['_meta']['width'][i])
        meta['height'] = int(sample[1]['_meta']['height'][i])
        meta['count'] = 3
        meta['crs'] = src_vrt.meta['crs']

        a = float(sample[1]['transform'][0][i])
        b = float(sample[1]['transform'][1][i])
        c = float(sample[1]['transform'][2][i])
        d = float(sample[1]['transform'][3][i])
        e = float(sample[1]['transform'][4][i])
        f = float(sample[1]['transform'][5][i])

        meta['transform'] = rio.Affine(a, b, c, d, e, f)

    #     saving the mask
        names = sample[2]
        output_filename = names[i].replace('.jp2', '.tif')   # change output format to tif as using GTiff driver

        outpath = os.path.join(path_to_save_mask, output_filename)

        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(_pred)