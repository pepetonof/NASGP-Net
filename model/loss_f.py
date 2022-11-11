# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:59:22 2021

@author: josef
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from scipy.spatial.distance import directed_hausdorff

#%%Dice Loss Class
"""Inspired by
    https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html#DiceLoss and
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth=1
    
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor):
        if not torch.is_tensor(inputs):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(inputs.shape))
        if not inputs.shape[-2:] == targets.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(inputs.shape, inputs.shape))
        if not inputs.device == targets.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    inputs.device, targets.device))
            
        # compute softmax over the classes axis
        inputs_soft = F.softmax(inputs, dim=1)
        
        # create the labels one hot tensor
        targets_one_hot = tgm.losses.one_hot(targets, num_classes=inputs.shape[1],
                                 device=inputs.device, dtype=inputs.dtype)
        
        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        cardinality = torch.sum(inputs_soft + targets_one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        return torch.mean(1. - dice_score)

#%%Dice Loss Function
def dice_loss(input:torch.Tensor, target:torch.Tensor):
    loss=DiceLoss()(input, target)
    return loss

#%%Combo Looss Class
"""Inspired by:
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Combo-Loss
    """
class ComboLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(ComboLoss, self).__init__()
        self.smooth=1
        self.alpha = alpha # < 0.5 penalises FP more, > 0.5 penalises FN more
        self.beta = beta #weighted contribution of modified CE loss compared to Dice loss
        self.eps=1e-3
    
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor):
        if not torch.is_tensor(inputs):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(inputs.shape))
        if not inputs.shape[-2:] == targets.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(inputs.shape, inputs.shape))
        if not inputs.device == targets.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    inputs.device, targets.device))
        
        # compute softmax over the classes axis
        inputs_soft = F.softmax(inputs, dim=1)
        #inputs_bool = (inputs_soft > 0.5).float()
        
        # create the labels one hot tensor
        targets_one_hot = tgm.losses.one_hot(targets, num_classes=inputs.shape[1],
                                 device=inputs.device, dtype=inputs.dtype)
        
        # compute the actual combo loss
        dims = (1, 2, 3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        cardinality = torch.sum(inputs_soft + targets_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        # bce
        inputs_soft = torch.clamp(inputs_soft, self.eps, 1.0 - self.eps) #Avoid extreme cases ln(0) and ln (1)
        out = -((self.beta * (targets_one_hot * torch.log(inputs_soft))) +
                                ((1-self.beta) * ((1-targets_one_hot) * torch.log(1 - inputs_soft))))
        
        # out = -(self.beta*targets_one_hot*torch.log(inputs_soft))+((1-self.beta)*(1.0-targets_one_hot)*torch.log(1.0 - inputs_soft))
        #print('CLOSS',out.shape) 
        nel=torch.numel(out)
        out = torch.sum(out, dims)
        weighted_ce = out/nel
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice)
        combo = torch.mean(combo)
        
        return combo + 0.5 #As offset because betewwn [-0.5, 0.5]

#%%Combo Function
def combo_loss(inputs:torch.Tensor, targets:torch.Tensor):
    Closs= ComboLoss(alpha=0.5, beta=0.7)#0.4
    #print(torch.max(targets), torch.min(targets))
    loss = Closs(inputs, targets)
    return loss

#%%IoU Class
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.smooth=1
        
    def forward(self, inputs, targets):
        if not torch.is_tensor(inputs):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(inputs.shape))
        if not inputs.shape[-2:] == targets.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(inputs.shape, inputs.shape))
        if not inputs.device == targets.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    inputs.device, targets.device))
        
        # compute softmax over the classes axis
        inputs_soft = F.softmax(inputs, dim=1)
        # # create the labels one hot tensor
        targets_one_hot = tgm.losses.one_hot(targets, num_classes=inputs.shape[1],
                                  device=inputs.device, dtype=inputs.dtype)
        # compute the actual IoU loss
        dims=(1,2,3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        cardinality = torch.sum(inputs_soft + targets_one_hot, dims)
        union = cardinality - intersection
        
        IoU = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - IoU #Loss form

#%%IoU function        
def IoU_loss(inputs:torch.Tensor, targets:torch.Tensor):
    return IoULoss()(inputs, targets)

#%%Hausdorff distance function
def hdistance_loss(inputs:torch.Tensor, targets:torch.Tensor):
    if not torch.is_tensor(inputs):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(inputs)))
    if not len(inputs.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(inputs.shape))
    if not inputs.shape[-2:] == targets.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                         .format(inputs.shape, inputs.shape))
    if not inputs.device == targets.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                inputs.device, targets.device))
    
    #Takes the placenta class
    inputs_soft = F.softmax(inputs, dim=1)
    sem_classes=['__background__', 'placenta']
    sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
    class_dim=1
    inputs_bool = inputs_soft.argmax(class_dim) == sem_class_to_idx['placenta']
    
    #Each element in batch
    hdistance=0.0
    for inp, tar in zip(inputs_bool,targets):
        hdistance += max(directed_hausdorff(inp.cpu().numpy(), tar.cpu().numpy())[0], 
                         directed_hausdorff(tar.cpu().numpy(), inp.cpu().numpy())[0])
    hdistance=hdistance/len(inputs)
    
    return hdistance
