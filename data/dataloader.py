# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 04:38:26 2023

@author: josef
"""


from torch.utils.data import DataLoader
import albumentations as A
from albumentations.augmentations.geometric.transforms import ElasticTransform
#from albumentations.augmentations.transforms import ElasticTransform
from albumentations.pytorch.transforms import ToTensorV2

from data.dataset import Dataset2D, Dataset3D, Dataset3D22D
import torch

class loaders:
    def __init__(self, train, valid, test, batch_size, 
                 image_height, image_width, 
                 # in_channels,
                 dataset_type=Dataset2D, no_classes_msk=2):
        
        self.TRAIN_IMG_DIR  = train["images"]
        self.VAL_IMG_DIR    = valid["images"]
        self.TEST_IMG_DIR   = test["images"]
        
        self.TRAIN_MASK_DIR = train["masks"]
        self.VAL_MASK_DIR   = valid["masks"]
        self.TEST_MASK_DIR  = test["masks"]
        
        self.BATCH_SIZE     = batch_size
        self.IMAGE_HEIGHT   = image_height
        self.IMAGE_WIDTH    = image_width
        self.NO_CLASSES_MSK   = no_classes_msk
        self.DATASET_TYPE = dataset_type
        
        
    def get_train_loader(self):
        train_transform = A.Compose(    [   
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                A.HorizontalFlip(p=0.5),
                ElasticTransform(alpha=1, sigma=10, alpha_affine=20, interpolation=1, 
                                  border_mode= 0, approximate=True, p=0.8),
                A.GridDistortion(num_steps=10, border_mode=0, p=0.5),
                A.Normalize(
                    mean=0.0,
                    std=1.0,
                    max_pixel_value=255.0,),
                # self.normalization_layer,
                ToTensorV2(transpose_mask=True),
            ],)
        
        #Dataset
        if self.DATASET_TYPE==Dataset2D:
            train_ds = self.DATASET_TYPE(self.TRAIN_IMG_DIR, 
                                         self.TRAIN_MASK_DIR, 
                                         transform=train_transform)
        
        elif self.DATASET_TYPE==Dataset3D22D:
            train_ds = Dataset3D(self.TRAIN_IMG_DIR, 
                                 self.TRAIN_MASK_DIR, 
                                 transform=train_transform,
                                 no_classes=self.NO_CLASSES_MSK)
            train_data = torch.cat([train_ds[i][0] for i in range(len(train_ds))], axis=0)
            mask_data = torch.cat([train_ds[i][1] for i in range(len(train_ds))], axis=0)
            
            # train_data = torch.unsqueeze(train_data, 1)
            # mask_data = torch.unsqueeze(mask_data, 1)
            
            train_ds = Dataset3D22D(train_data, mask_data)
        
        elif self.DATASET_TYPE==Dataset3D:
            train_ds = self.DATASET_TYPE(self.TRAIN_IMG_DIR, 
                                         self.TRAIN_MASK_DIR, 
                                         transform=train_transform,
                                         no_classes=self.NO_CLASSES_MSK)
        #DataLoader
        train_loader = DataLoader(train_ds,
                                  batch_size=self.BATCH_SIZE,
                                  shuffle=False,
                                  )
        
        return train_loader, train_ds
    
    
    def get_val_loader(self):
        valid_transform = A.Compose(    [   
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                A.Normalize(
                    mean=0.0,
                    std=1.0,
                    max_pixel_value=255.0,),
                # self.normalization_layer,
                ToTensorV2(transpose_mask=True),
            ],)
        
        #Dataset
        if self.DATASET_TYPE==Dataset2D:
            val_ds = self.DATASET_TYPE(self.VAL_IMG_DIR, 
                                         self.VAL_MASK_DIR,  
                                         transform=valid_transform)
        
        elif self.DATASET_TYPE==Dataset3D22D:
            val_ds = Dataset3D(self.VAL_IMG_DIR, 
                                         self.VAL_MASK_DIR,  
                                         transform=valid_transform,
                                         no_classes=self.NO_CLASSES_MSK)
            val_data = torch.cat([val_ds[i][0] for i in range(len(val_ds))], axis=0)
            mask_data = torch.cat([val_ds[i][1] for i in range(len(val_ds))], axis=0)
            # print(val_data.shape)
            # print(mask_data.shape)
            val_ds = self.DATASET_TYPE(val_data, mask_data)
        
        elif self.DATASET_TYPE==Dataset3D:
            val_ds = self.DATASET_TYPE(self.VAL_IMG_DIR, 
                                         self.VAL_IMG_DIR, 
                                         transform=valid_transform,
                                         no_classes=self.NO_CLASSES_MSK)
                
        #DataLoader
        val_loader = DataLoader(val_ds,
                                batch_size=1,#self.BATCH_SIZE,
                                shuffle=False,
                                )
        
        return val_loader, val_ds
    
    
    def get_test_loader(self):
        test_transform = A.Compose(    [   
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                A.Normalize(
                    mean=0.0,
                    std=1.0,
                    max_pixel_value=255.0,),
                # self.normalization_layer,
                ToTensorV2(transpose_mask=True),
            ],)
        
        #Dataset
        if self.DATASET_TYPE==Dataset2D:
            test_ds = self.DATASET_TYPE(self.TEST_IMG_DIR, 
                                         self.TEST_MASK_DIR,  
                                         transform=test_transform)
        
        elif self.DATASET_TYPE==Dataset3D22D:
            test_ds = Dataset3D(self.TEST_IMG_DIR, 
                                self.TEST_MASK_DIR,  
                                transform=test_transform,
                                no_classes=self.NO_CLASSES_MSK)
            test_data = torch.cat([test_ds[i][0] for i in range(len(test_ds))], axis=0)
            mask_data = torch.cat([test_ds[i][1] for i in range(len(test_ds))], axis=0)
            test_ds = self.DATASET_TYPE(test_data, mask_data)
        
        elif self.DATASET_TYPE==Dataset3D:
            test_ds = self.DATASET_TYPE(self.TEST_IMG_DIR, 
                                         self.TEST_IMG_DIR, 
                                         transform=test_transform,
                                         no_classes=self.NO_CLASSES_MSK)
                
        #DataLoader
        test_loader = DataLoader(test_ds,
                                 batch_size=1,
                                 shuffle=False,
                                )
        
        return test_loader, test_ds