# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:38:53 2023

@author: josef
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.Surface_distance_based_measures import (compute_surface_distances, 
                                             compute_surface_dice_at_tolerance,
                                             compute_robust_hausdorff,
                                             compute_dice_coefficient)

def one_hot(labels, num_classes, device, dtype):
    shape = labels.shape
    # print('OneHot', labels.shape, labels.device, dtype, device)
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

#%%Dice Metric
#Dice Score using tensors use Arg max instead of softmax in loss dice 
#similar to MOANI and Kornia Dice Metric implementation
class DiceMetric(nn.Module):
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
    
    #One Hot Format
    #input (batch size, num_clases, *spatial dims) after softmax?
    #target (batch size, num_clases, *spatial dim)
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        num_classes = _input.shape[1]
        dims = torch.arange(2, len(_input.shape)).tolist()
        if self.average=='micro':
            dims = (1, *dims)

        if self.softmax:
            _input = F.softmax(_input, dim=1)
        _input = torch.argmax(_input, dim=1, keepdim=True)

        _input = one_hot(_input.squeeze(1), num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        _target = one_hot(_target, num_classes=num_classes, device=_input.device, dtype=_input.dtype)

        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]

        intersection = torch.sum(_input * _target, dims)
        cardinality = torch.sum(_input + _target, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        
        dice_score = torch.mean(dice_score)

        return dice_score.detach().item()
    
#%%IoU Metric
class IoUMetric(nn.Module):
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
    
    #One Hot Format
    #input (batch size, num_clases, *spatial dims) after softmax?
    #target (batch size, num_clases, *spatial dim)
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        num_classes = _input.shape[1]
        dims = torch.arange(2, len(_input.shape)).tolist()
        if self.average=='micro':
            dims = (1, *dims)

        if self.softmax:
            _input = F.softmax(_input, dim=1)
        _input = torch.argmax(_input, dim=1, keepdim=True)

        _input = one_hot(_input.squeeze(1), num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        _target = one_hot(_target, num_classes=num_classes, device=_input.device, dtype=_input.dtype)

        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]

        intersection = torch.sum(_input * _target, dims)
        cardinality = torch.sum(_input + _target, dims)
        union = cardinality-intersection
        iou = (intersection + self.eps) / (union + self.eps)
        
        return torch.mean(iou).detach().item()

#%%Hausdorff distance Metric
class HDMetric(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool=True,
                 spacing_mm:tuple=(1,1,1),) -> None:
        super().__init__()
        self.average = average
        self.include_background=include_background
        self.softmax=softmax
        self.spacing_mm=spacing_mm
    
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        num_classes = _input.shape[1]
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
            
        _input = torch.argmax(_input, dim=1, keepdim=True)
        
        _input = one_hot(_input.squeeze(1), num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        _target = one_hot(_target, num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
            num_classes =- 1

        hds=[]
        hds95=[]
        
        hds_c=[]
        hds95_c=[]
        
        for c in range(_input.shape[1]):  
            for b in range(_input.shape[0]):
                _inputb = _input[b, c]
                _targetb = _target[b, c]
                
                if len(_inputb.shape)==2 and len(_targetb.shape)==2:
                    _inputb = torch.unsqueeze(_inputb, 2)
                    _targetb = torch.unsqueeze(_targetb, 2)

                surface_distances = compute_surface_distances(_targetb.cpu().numpy(), _inputb.cpu().numpy(), self.spacing_mm)
                hd = compute_robust_hausdorff(surface_distances, 100)
                hd95 = compute_robust_hausdorff(surface_distances, 95)
                # hd95 = compute_robust_hausdorff(surface_distances, 95)
                hds.append(hd)
                hds95.append(hd95)
   
            if self.average=='macro':
                hds_c.append(torch.mean(torch.tensor(hds)))
                hds95_c.append(torch.mean(torch.tensor(hds95)))
                hds=[]
                hds95=[]

        if self.average=='micro':
            return torch.mean(torch.tensor(hds)).detach().item(), torch.mean(torch.tensor(hds95)).detach().item()
        elif self.average=='macro':
            return torch.mean(torch.tensor(hds_c)).detach().item(), torch.mean(torch.tensor(hds95_c)).detach().item()
        
#%%Surface Dice pr Normalised Surface Distance (NSD)
class NSDMetric(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool=True,
                 spacing_mm:tuple=(1,1,1),
                 tolerance:int = 1) -> None:
        super().__init__()
        self.average = average
        self.include_background=include_background
        self.softmax=softmax
        self.spacing_mm=spacing_mm
        self.tolerance = tolerance
    
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        num_classes = _input.shape[1]
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
        
        _input = torch.argmax(_input, dim=1, keepdim=True)
        
        _input = one_hot(_input.squeeze(1), num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        _target = one_hot(_target, num_classes=num_classes, device=_input.device, dtype=_input.dtype)
        
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
            num_classes = num_classes- 1

        nsds=[]
        nsds_c = []
        
        for c in range(_input.shape[1]):  
            for b in range(_input.shape[0]):
                _inputb = _input[b, c]
                _targetb = _target[b, c]
                
                if len(_inputb.shape)==2 and len(_targetb.shape)==2:
                    _inputb = torch.unsqueeze(_inputb, 2)
                    _targetb = torch.unsqueeze(_targetb, 2)
                
                # print(_inputb.shape, _targetb.shape)

                surface_distances = compute_surface_distances(_targetb.cpu().numpy(), _inputb.cpu().numpy(), self.spacing_mm)
                nsd = compute_surface_dice_at_tolerance(surface_distances, self.tolerance)
                # hd95 = compute_robust_hausdorff(surface_distances, 95)
                nsds.append(nsds)
                
            if self.average=='macro':
                nsds_c.append(torch.mean(torch.tensor(nsd)))
                nsds=[]

        if self.average=='micro':
            return torch.mean(torch.tensor(nsds)).detach().item()
        elif self.average=='macro':
            return torch.mean(torch.tensor(nsds_c)).detach().item()