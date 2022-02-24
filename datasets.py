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

import glob
import pandas as pd

######

class TestDrivewayDataset(Dataset):
    def __init__(self, df, config):
        
        self.config = config
        self.df = df
        
#         self.reshape = self.config['reshape']
        self.reshape_size = self.config['reshape_size']
        
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return(len(self.df))
    
    def transform(self, img, mask_building, mask_road):
        """
        Input Tensor, Ouput Tensor
        
        """
        
        # To Tensor
        img = torch.tensor(img)
        mask_building = torch.tensor(mask_building)
        mask_road = torch.tensor(mask_road)
        
        # To PIL image
        image = self.to_pil(img[:3])
        mask_building = self.to_pil(mask_building)
        mask_road = self.to_pil(mask_road)
        
        # Resize
        image = TF.resize(image, size = self.reshape_size, interpolation=PIL.Image.NEAREST)
        mask_building = TF.resize(mask_building, size = self.reshape_size, interpolation=PIL.Image.NEAREST)
        mask_road = TF.resize(mask_road, size = self.reshape_size, interpolation=PIL.Image.NEAREST)
        
        # Change to tensors
        image = self.to_tensor(image)
        mask_building = self.to_tensor(mask_building)
        mask_road = self.to_tensor(mask_road)
        
        # Merge input tensors to 5 channel, 3 image 1 building 1 road
        _input_stacked = torch.cat((image, mask_building, mask_road))
        
        return _input_stacked
    
    def morph_meta(self, meta):
        _meta = meta

        affine = _meta['transform']
        affine_trans = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f,]
                
        _meta.pop('transform') # removed as it might crash the program, might be fixed using a custom collate.
        _meta.pop('nodata')    # removed as it might crash the program, might be fixed using a custom collate.
        
        meta_dict = {'_meta': _meta,
                     'transform': affine_trans}
        
        return meta_dict
        
    def __getitem__(self, index):
                
        # prepapre reading paths
        path_img = self.df.iloc[index]['path_img']
        path_building = self.df.iloc[index]['path_building_fp']
        path_road = self.df.iloc[index]['path_road_mask']
        
        # read src datasets
        src_img = rio.open(path_img, mode = 'r')
        src_building = rio.open(path_building, mode ='r')
        src_road = rio.open(path_road, mode = 'r')
        
        meta = src_img.meta
        
        # read images, if you get nullpointer error in img read, there is a channel issue. Reformat the images.
        _img = src_img.read()
        _mask_building = src_building.read(1)
        _mask_road = src_road.read(1)
        
        _input_tensor = self.transform(_img, _mask_building, _mask_road)
        meta_dict = self.morph_meta(src_img.meta)
        name = path_img.split('/')[-1]
        
        return [_input_tensor, meta_dict, name]