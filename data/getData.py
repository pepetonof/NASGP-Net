# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 08:43:56 2023

@author: josef
"""

import albumentations as A
from albumentations.augmentations.geometric.transforms import ElasticTransform
#from albumentations.augmentations.transforms import ElasticTransform
from albumentations.pytorch.transforms import ToTensorV2
from data.dataset import DataSet

class loaders2():
    def __init__(self, batch_size, 
                 image_height, image_width, 
                 in_channels, out_channels,):
                 #num_workers=2, pin_memory=True):
        
        self.BATCH_SIZE     = batch_size
        self.IMAGE_HEIGHT   = image_height
        self.IMAGE_WIDTH    = image_width
        self.IN_CHANNELS    = in_channels
        self.OUT_CHANNELS   = out_channels        
        
        if self.IN_CHANNELS==3:
            self.normalization_layer = A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,)
        elif self.IN_CHANNELS==1:
            self.normalization_layer = A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,)
    
    def get_dataset(self, images_dir, masks_dir, train=True):
        if train == True:
            transform = A.Compose([# A.ToGray(p=1.0),
                                   # A.Equalize(p=1.0),
                                   A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                                   A.HorizontalFlip(p=0.5),
                                   ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
                                                    border_mode= 0, approximate=True, p=0.8),
                                   A.GridDistortion(num_steps=10, border_mode=0, p=0.5),
                                   self.normalization_layer,
                                   ToTensorV2(),
                                   ],)
        else:
            transform = A.Compose([A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                                   self.normalization_layer,
                                   ToTensorV2(),
                                   ],)
            
        dataset = DataSet(image_dir= images_dir,
                          mask_dir = masks_dir,
                          transform= transform,
                          no_classes = self.OUT_CHANNELS)
        
        return dataset