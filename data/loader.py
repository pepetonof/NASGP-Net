# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:09:07 2021

@author: josef

Get validation and train loader. Validation and train loader are not always
same size
"""
from torch.utils.data import DataLoader
# from pathlib import Path
import albumentations as A
from albumentations.augmentations.geometric.transforms import ElasticTransform
#from albumentations.augmentations.transforms import ElasticTransform
from albumentations.pytorch.transforms import ToTensorV2
from data.dataset import DataSet

class loaders():
    def __init__(self, train, valid, test, batch_size, 
                 image_height, image_width, in_channels,
                 num_workers=2, pin_memory=True):
        
        self.TRAIN_IMG_DIR  = train["images"]
        self.VAL_IMG_DIR    = valid["images"]
        self.TEST_IMG_DIR   = test["images"]
        
        self.TRAIN_MASK_DIR = train["masks"]
        self.VAL_MASK_DIR   = valid["masks"]
        self.TEST_MASK_DIR  = test["masks"]
        
        self.BATCH_SIZE     = batch_size
        self.IMAGE_HEIGHT   = image_height
        self.IMAGE_WIDTH    = image_width
        self.IN_CHANNELS    = in_channels
        self.NUM_WORKERS    = num_workers
        self.PIN_MEMORY     = pin_memory
        
        
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
        
    def get_train_loader(self):
        train_transform = A.Compose(
            [   
                # A.ToGray(p=1.0),
                # A.Equalize(p=1.0),
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                A.HorizontalFlip(p=0.5),
                ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
                                 border_mode= 0, approximate=True, p=0.8),
                A.GridDistortion(num_steps=10, border_mode=0, p=0.5),
                self.normalization_layer,
                ToTensorV2(),
            ],)
        train_ds = DataSet(
            image_dir=self.TRAIN_IMG_DIR,
            mask_dir=self.TRAIN_MASK_DIR,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=True,
        )
        return train_loader, train_ds,
    
    def get_val_loader(self, num_workers=2):
        val_transform = A.Compose(
                        [   
                            # A.ToGray(p=1.0),
                            # A.Equalize(p=1.0),
                            A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                            self.normalization_layer,
                            ToTensorV2(),
                        ],)
        val_ds = DataSet(
            image_dir=self.VAL_IMG_DIR,
            mask_dir=self.VAL_MASK_DIR,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=False,
        )
        return val_loader, val_ds
    
    def get_test_loader(self, num_workers=2):
        test_transform = A.Compose(
                        [   
                            # A.ToGray(p=1.0),
                            # A.Equalize(p=1.0),
                            A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                            self.normalization_layer,
                            ToTensorV2(),
                        ],)
        test_ds = DataSet(
            image_dir=self.TEST_IMG_DIR,
            mask_dir=self.TEST_MASK_DIR,
            transform=test_transform,
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            pin_memory=self.PIN_MEMORY,
            shuffle=False,
        )
        return test_loader, test_ds
