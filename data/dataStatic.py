# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:10:30 2022

@author: josef
"""
import os
from pathlib import Path

def get_data(p):
    p=Path(p)
    if not os.path.exists(p):
        raise ValueError('Debe existir el directorio')
        
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    dirs.sort(key=lambda d: d.name)
    dirs=dirs[2:]#!!!
    
    train=[x for x in dirs[[d.name for d in dirs].index('train')].iterdir() if x.is_dir()]
    train.sort(key=lambda d: d.name)
    
    valid=[x for x in dirs[[d.name for d in dirs].index('valid')].iterdir() if x.is_dir()]
    valid.sort(key=lambda d: d.name)
    
    test=[x for x in dirs[[d.name for d in dirs].index('test')].iterdir() if x.is_dir()]
    test.sort(key=lambda d: d.name)

    TRAIN_IMG_DIR = list(train[0].glob('**/*.png'))
    TRAIN_MASK_DIR= list(train[1].glob('**/*.png'))
    VAL_IMG_DIR   = list(valid[0].glob('**/*.png'))
    VAL_MASK_DIR  = list(valid[1].glob('**/*.png'))
    TEST_IMG_DIR  = list(test[0].glob('**/*.png'))
    TEST_MASK_DIR = list(test[1].glob('**/*.png'))
    
    train = dict(images=TRAIN_IMG_DIR, masks=TRAIN_MASK_DIR)
    valid = dict(images=VAL_IMG_DIR, masks=VAL_MASK_DIR)
    test  = dict(images=TEST_IMG_DIR, masks=TEST_MASK_DIR)
    
    return train, valid, test