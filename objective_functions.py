# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:58:29 2023

@author: josef
"""

import torch.optim as optim
from model.train_valid import train_and_validate
from model.predict import test

#Version of evaluate for surrogate model
def evaluate_NoParameters(model, in_channels, max_params, pset):
    """Make model"""
    # model = make_model(ind, in_channels, pset)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return (max_params - params)/max_params, params

def evaluate_Segmentation(model, nepochs, lossfn, lr, 
                          loaders, device, ruta, verbose_train):
    
    """Train, val and test loaders"""
    train_loader, _ = loaders.get_train_loader() #loaders.get_train_loader(288, 480)
    val_loader, _  = loaders.get_val_loader()
    test_loader, _ = loaders.get_test_loader()
    
    # """Make model"""
    # in_channels = loaders.IN_CHANNELS
    # model = make_model(ind, in_channels, pset)
    
    """Hyoperparameters for train"""
    LOAD_MODEL = False
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    """Save model and images"""
    save_model=False
    save_images=False
    
    """Train and valid model"""
    train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
        model, train_loader, val_loader,
        nepochs, optimizer, lossfn,
        device, LOAD_MODEL, save_model,
        ruta=ruta, verbose=verbose_train
        )
    
    """Test model"""
    dices, ious, hds = test(test_loader, model, lossfn,
                            save_imgs=save_images, ruta=ruta, device=device, verbose=verbose_train)
    
    metrics={}
    train_valid={}
    
    train_valid["train_loss"]=train_loss
    train_valid["valid_loss"]=valid_loss
    train_valid["train_dice"]=train_dice
    train_valid["valid_dice"]=valid_dice
    
    #List of the metric reached on each image in the test dataset
    metrics["dices"]=dices
    metrics["ious"]=ious
    metrics["hds"]=hds
    
    return metrics, train_valid