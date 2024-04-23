# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:10:30 2022

@author: josef
"""
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_data(p, val_size=0.1, _format='.png'):
    p=Path(p)
    if not os.path.exists(p):
        raise ValueError('Debe existir el directorio')
        
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    dirs.sort(key=lambda d: d.name)
    # dirs=dirs[2:]#!!!
    # print(dirs)
    
    train=[x for x in dirs[[d.name for d in dirs].index('train')].iterdir() if x.is_dir()]
    # print(type(train))
    train.sort(key=lambda d: d.name)
    
    test=[x for x in dirs[[d.name for d in dirs].index('test')].iterdir() if x.is_dir()]
    test.sort(key=lambda d: d.name)
    
    TRAIN_IMG_DIR = list(train[0].glob('**/*'+_format))
    TRAIN_MASK_DIR= list(train[1].glob('**/*'+_format))
    
    TEST_IMG_DIR  = list(test[0].glob('**/*'+_format))
    TEST_MASK_DIR = list(test[1].glob('**/*'+_format))
    
    if 'valid' in [d.name for d in dirs]:
        valid=[x for x in dirs[[d.name for d in dirs].index('valid')].iterdir() if x.is_dir()]
        valid.sort(key=lambda d: d.name)
        
        VAL_IMG_DIR   = list(valid[0].glob('**/*'+_format))
        VAL_MASK_DIR  = list(valid[1].glob('**/*'+_format))
        
    #Infer valid from train data
    else:
        TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            test_size = val_size,
            random_state=0,
            shuffle=False
            )

    train = dict(images=TRAIN_IMG_DIR, masks=TRAIN_MASK_DIR)
    valid = dict(images=VAL_IMG_DIR, masks=VAL_MASK_DIR)
    test  = dict(images=TEST_IMG_DIR, masks=TEST_MASK_DIR)
    
    return train, valid, test


##Always generate a validation set from the train set depending on the val_size parameter
def get_data_folds(p, val_size=0.1, _format='.png'):
    p = Path(p)
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    dirs.sort(key=lambda d: int(d.name))
    
    # print('DIRS', dirs)
    train_folds=[]
    valid_folds=[]
    test_folds=[]
    
    # print(dirs)
    for fold in dirs:
        folders = [x for x in fold.iterdir() if x.is_dir()]
        
        train = [x for x in folders[[d.name for d in folders].index('train')].iterdir() if x.is_dir()]
        train.sort(key=lambda d:d.name)
        
        test =  [x for x in folders[[d.name for d in folders].index('test')].iterdir() if x.is_dir()]
        test.sort(key=lambda d:d.name)
        
        # print(len(train), len(test))
        # print(train[0], train[1])
        # print(test[0], test[1])
        
        # TRAIN_IMG_DIR = list(train[0].glob('**/*'+_format))
        # TRAIN_MASK_DIR= list(train[1].glob('**/*'+_format))
        
        files_inp = list(train[0].glob('**/*'+_format))
        files_msk = list(train[1].glob('**/*'+_format))
        
        # print(len(files_inp), len(files_msk))
        
        TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
            files_inp,
            files_msk,
            test_size = val_size,
            random_state=0,
            shuffle=False
            )
        
        TEST_IMG_DIR  = list(test[0].glob('**/*'+_format))
        TEST_MASK_DIR = list(test[1].glob('**/*'+_format))
        
        # print('TRAIN',len(TRAIN_IMG_DIR), len(TRAIN_MASK_DIR))
        # print('VAL',len(VAL_IMG_DIR), len(VAL_MASK_DIR))
        # print('TEST',len(TEST_IMG_DIR), len(TEST_MASK_DIR))
        
        # print()

        
        train = dict(images=TRAIN_IMG_DIR, masks=TRAIN_MASK_DIR)
        valid = dict(images=VAL_IMG_DIR, masks=VAL_MASK_DIR)
        test  = dict(images=TEST_IMG_DIR, masks=TEST_MASK_DIR)
        
        train_folds.append(train)
        valid_folds.append(valid)
        test_folds.append(test)
        
        
        # test = [x for x in dirs[[d.name for d in dirs].index('test')].iterdir() if x.is_dir()]
        # train = [x for x in dirs[[d.name for d in dirs].index('train')].iterdir() if x.is_dir()]
        # print('TRAIN', train)
        # print('TEST', test)
        
        
        # break
    return train_folds,valid_folds, test_folds
    

# path_images = 'C:/Users/josef/serverBUAP/datasets/images_PROMISE'
# get_data(path_images)
# print()

# path_images = "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/comparison-datasets/folds"
# train, valid, test = get_data_folds(path_images, _format='.jpg') #train, valid, test = 
# print(len(train))
# print(len(test))
# print(len(valid))

