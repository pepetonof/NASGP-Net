# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:58:29 2023

@author: josef
"""

import torch.optim as optim
from model.train_valid import train_and_validate
from model.predict import test

#Version of evaluate for surrogate model
def evaluate_NoParameters(model, max_params):
    """Make model"""
    # model = make_model(ind, in_channels, pset)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return (max_params - params)/max_params, params

#fold only for print purposes
def evaluate_Segmentation(model, num_epochs, tolerance, loss_fn, metrics, lr, 
                          loaders, device, ruta, verbose_train,
                          save_model, save_images, fold=None):
    
    """Train, val and test loaders"""
    train_loader, _ = loaders.get_train_loader() #loaders.get_train_loader(288, 480)
    val_loader, _  = loaders.get_val_loader()
    test_loader, _ = loaders.get_test_loader()
    
    """Hyoperparameters for train"""
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    """Train and valid model"""
    lossAndDice, metricsVal = train_and_validate(model, train_loader, val_loader,
                                                 num_epochs, tolerance,
                                                 optimizer, loss_fn, metrics,
                                                 device, save_model, save_images,
                                                 ruta=ruta, verbose=verbose_train, fold=fold)
                                                
    """Test model"""
    metricsTest = test(test_loader, model, metrics,
                       save_imgs=save_images, ruta=ruta, 
                       device=device, verbose=verbose_train, fold=fold)
    
    return metricsTest, metricsVal, lossAndDice