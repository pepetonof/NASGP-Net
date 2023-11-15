# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:38:53 2023

@author: josef
"""
from model.loss_f import DiceLoss, IoULoss

def dice_score(input, target):
    diceloss = DiceLoss(average='micro', include_background=True, eps=1.0)
    dicescore = 1.0 - diceloss(input, target)
    
    return dicescore.detach().item()

def iou_score(input, target):
    iouloss = IoULoss(average='micro', include_background=True, eps=1.0)
    iouscore = 1.0 - iouloss(input, target)
    
    return iouscore.detach().item()