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
import numpy as np

#%%Dice Loss
#https://docs.monai.io/en/0.3.0/_modules/monai/metrics/meandice.html#compute_meandice
#https://stats.stackexchange.com/questions/298849/soft-version-of-the-maximum-function#:~:text=softmax%20is%20a%20smooth%20approximation%20of%20the%20argmax,max%27s%20index%2C%20as%20opposed%20to%20an%20ordinal%20position%29.
class DiceLoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool=True,
                 eps:float=1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.include_background=include_background
        self.softmax=softmax

    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        if not len(_input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {_input.shape}")

        if not _input.shape[-2:] == _target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {_input.shape} and {_target.shape}")

        if not _input.device == _target.device:
            raise ValueError(f"input and target must be in the same device. Got: {_input.device} and {_target.device}")
        
        dims = torch.arange(2, len(_input.shape)).tolist()
        #'macro':Calculate the metric for each class separately, and average the metrics across classes
        if self.average == "micro": #Calculate the metric globally, across all samples and classes.
            dims = (1, *dims)
        
        
        # print('DiceLoss Input', _input.shape, _target.shape)
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
            
        # print('DiceLoss Softmax', _input.shape, _target.shape)
        
        # create the labels one hot tensor
        
        # c = F.one_hot(_target, 2)
        # torch.squeeze(torch.transpose(c.unsqueeze(1), 1, 4), 4)
        _target = tgm.losses.one_hot(_target, num_classes=_input.shape[1],
                                 device=_input.device, dtype=_input.dtype)
        
        # print("DiceLoss One Hot", _input.shape, _target.shape).
        
        #Ignore background?
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
        
        # print("Shape", input_soft.shape, target_one_hot.shape)
        
        # compute the actual dice score
        intersection = torch.sum(_input * _target, dims)
        
        # print("\nIntersection \t", intersection, intersection.shape)
        cardinality = torch.sum(_input + _target, dims)
        # print("Cardinality \t", cardinality, cardinality.shape)
        
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        
        # print("DiceScore:\t", dice_score, dice_score.shape)
        
        dice_loss = -dice_score + 1.0
        
        # print("Dice_loss \t", dice_loss, dice_loss.shape)

        # reduce the loss across samples (and classes in case of `macro` averaging)
        # print("\nDice Mean4Ch:\t", torch.mean(dice_loss, axis=0))
        dice_loss = torch.mean(dice_loss)#, axis=0)
        # print("Dice Mean:\t", dice_loss, dice_loss.shape)
        # dice_loss = torch.mean(dice_loss[1:])
        # print("Average Dice Loss", dice_loss)
        
        return dice_loss

#%%Jaccard Index - IoU Loss
class IoULoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool = True,
                 eps:float=1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.softmax = softmax,
        self.include_background=include_background

    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        
        if not len(_input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW or BxNxHxWxD. Got: {_input.shape}")

        if not _input.shape[-2:] == _target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {_input.shape} and {_target.shape}")

        if not _input.device == _target.device:
            raise ValueError(f"input and target must be in the same device. Got: {_input.device} and {_target.device}")
        
        dims = torch.arange(2, len(_input.shape)).tolist()
        if self.average=='micro':
            dims = (1, *dims)
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
        
        # create the labels one hot tensor
        _target = tgm.losses.one_hot(_target, num_classes=_input.shape[1],
                                 device=_input.device, dtype=_input.dtype)
        
        #Ignore background?
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
        
        
        # compute the actual iou score
        intersection = torch.sum(_input * _target, dims)
        cardinality = torch.sum(_input + _target, dims)
        union = cardinality - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = -iou + 1.0
        
        # reduce the loss across samples (and classes in case of `macro` averaging)
        iou_loss = torch.mean(iou_loss)#, axis=0)
        
        return iou_loss

#%%Combo Loss
#From https://github.com/anwai98/Loss-Functions/blob/main/loss-function-library-keras-pytorch.ipynb
class ComboLoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool=False,
                 eps:float=1e-8,
                 alpha:float=0.5,
                 beta:float=0.5) -> None:
        self.average = average
        self.eps = eps
        self.softmax = softmax,
        self.include_background=include_background
        self.alpha=alpha
        self.beta=beta
        
    
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        if not len(_input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW or BxNxHxWxD. Got: {input.shape}")

        if not _input.shape[-2:] == _target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {_input.shape} and {_target.shape}")

        if not _input.device == _target.device:
            raise ValueError(f"input and target must be in the same device. Got: {_input.device} and {_target.device}")
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
        
        # create the labels one hot tensor
        _target = tgm.losses.one_hot(_target, num_classes=_input.shape[1],
                                 device=_input.device, dtype=_input.dtype)
        
        #Ignore background?
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
        
        #Set diomension for the appropiate averaging
        dims: tuple[int, ...] = (2, 3)
        if self.average == "micro":
            dims = (1, *dims)
        
        # dice_sore
        intersection = torch.sum(_input * _target, dims)
        cardinality = torch.sum(_input + _target, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        
        # bce
        _input = torch.clamp(_input, self.eps, 1.0 - self.eps) #Avoid extreme cases ln(0) and ln (1)
        out = -((self.beta * (_target * torch.log(_input))) +
                                ((1-self.beta) * ((1-_target) * torch.log(1 - _input))))
        
        nel=torch.numel(out)
        out = torch.sum(out, dims)
        weighted_ce = out/nel
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice_score)
        combo = torch.mean(combo)
        
        # print("\nMyCombo", combo, combo.shape, dice, weighted_ce)
        
        return combo + 0.5
    
#%%Hausdorff Distance Loss
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
    hd=0.0
    hd95=0.0
    
    for inp, tar in zip(inputs_bool,targets):
        hd1 = directed_hausdorff(inp.cpu().numpy(), tar.cpu().numpy())[0]
        hd2 = directed_hausdorff(tar.cpu().numpy(), inp.cpu().numpy())[0]
        
        # print(type(hd1), type(hd2), hd1, hd2)
        
        hd += max(hd1, hd2)
        hd95 += np.percentile(np.hstack((hd1, hd2)), 95)
        # hdistance += max(directed_hausdorff(inp.cpu().numpy(), tar.cpu().numpy())[0], 
        #                  directed_hausdorff(tar.cpu().numpy(), inp.cpu().numpy())[0])
        
    hd=hd/len(inputs)
    hd95=hd95/len(inputs)
    
    return hd, hd95