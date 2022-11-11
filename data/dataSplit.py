# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 07:51:38 2021

@author: josef
"""
##Para lecutra de archivos localmente
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_ind(p:Path()):
    string=os.path.basename(p)
    num=""
    for i in string:
        if i.isdigit():
            num=num+i  
    return int(num)

#Se toma en cuenta conjunto de entrenamiento, validación y prueba
def get_data(train_size, val_size, test_size, p):
    p=Path(p)
    if not os.path.exists(p):
        raise ValueError('Debe existir el directorio')
    if train_size + val_size + test_size != 1.0:
        raise ValueError('La suma de los tamaños debe ser igual a 1')
    
    dirs=[x for x in p.iterdir() if x.is_dir()] #Directorios
    dirs.sort(key=lambda d: d.name)
    dirs=dirs[:2]##!!!

    files_inp=list(dirs[0].glob('**/*.png')) #Images input
    files_msk=list(dirs[1].glob('**/*.png')) #Images output
    
    #Ordering according to index in tittle image
    files_inp.sort(key=get_ind)
    files_msk.sort(key=get_ind)

    random_seed=0
    
    #Test data
    x_remain, TEST_IMG_DIR, y_remain, TEST_MASK_DIR = train_test_split(
        files_inp,
        files_msk,
        test_size=test_size,
        random_state=random_seed,
        shuffle=False
        )
    
    #Adjust val_size, train_size
    remain_size = 1.0 - test_size
    val_size_adj =val_size / remain_size
    
    TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR = train_test_split(
        x_remain,
        y_remain, 
        train_size = 1-val_size_adj,
        random_state = random_seed,
        shuffle=False
        )
    
    train = dict(images=TRAIN_IMG_DIR, masks=TRAIN_MASK_DIR)
    valid = dict(images=VAL_IMG_DIR, masks=VAL_MASK_DIR)
    test  = dict(images=TEST_IMG_DIR, masks=TEST_MASK_DIR)
    
    return train, valid, test
    
