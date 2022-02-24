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
import argparse

# Initilization of arguments
parser = argparse.ArgumentParser(description='Provide city name to process.')
parser.add_argument('--path_to_tiles', type=str, required=True, help='input image tiles')
parser.add_argument('--path_to_roads', type=str, required=True, help='input road masks')
parser.add_argument('--path_to_buildings', type=str, required=True, help='input building masks')
parser.add_argument('--path_to_model', type=str, required=True, help='model to run')
parser.add_argument('--path_to_save_mask', type=str, required=True, help='path to save the generated masks')
parser.add_argument('--model_trained_multi_gpu', type=bool, required=True, help='was model trained on multiple gpus')
parser.add_argument('--tile_resize_dim', type=int, required=False, default=768, help='path to save the generated masks')
args = parser.parse_args()


path_imgs = args.path_to_tiles
path_road_masks = args.path_to_roads
path_building_fps = args.path_to_buildings
path_to_model = args.path_to_model
path_to_save_mask = args.path_to_save_mask

MODEL_TRAINED_ON_MULTIPLE_GPU = args.model_trained_multi_gpu
RESHAPE_SIZE = args.tile_resize_dim

# sorting to match files
path_imgs.sort()
path_road_masks.sort()
path_building_fps.sort()

BATCH_SIZE = 12
NUM_WORKERS = 12

df = pd.DataFrame(columns = ['path_img', 'path_road_mask', 'path_building_fp'])


def add_entry(df_, path_imgs, path_road_masks, path_building_fps):
    """
    Add a list of img paths and sw_mask paths to the dataframe.
    In future you may change it to generalize using zip(input_values) to form a new row rather than hardcoding.
    """
    for path_img, path_road_mask, path_building_fp in zip(path_imgs, 
                                                          path_road_masks, 
                                                          path_building_fps):
        new_row = {'path_img': path_img,
                   'path_road_mask': path_road_mask,
                   'path_building_fp': path_building_fp
                  }
        df_ = df_.append(new_row, ignore_index=True)
    return df_


df = add_entry(df, path_imgs, path_road_masks, path_building_fps)

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

        # adding functionality to convert the file to .jp2 and delete the .tif
        infile_path = outpath # path to the saved .tif file
        outfile_path = os.path.join(path_to_save_mask, names[i]) # path to the saved .jp2 file

        # adding non-blocking functionality
        p = subprocess.Popen(["./tiff2jp2.sh",infile_path,outfile_path])