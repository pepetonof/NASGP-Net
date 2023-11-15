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

# from skimage.segmentation import find_boundaries
# from scipy.ndimage import distance_transform_edt


#https://docs.monai.io/en/0.3.0/_modules/monai/metrics/meandice.html#compute_meandice
class DiceLoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 eps:float=1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.include_background=include_background

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
        
        input_soft = F.softmax(input, dim=1)
        
        # create the labels one hot tensor
        target_one_hot = tgm.losses.one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        
        # print("Shape", input_soft.shape, target_one_hot.shape)
        
        #Ignore background?
        if not self.include_background:
            input_soft = input_soft[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        
        # print("Shape", input_soft.shape, target_one_hot.shape)
        
        #Set diomension for the appropiate averaging
        dims: tuple[int, ...] = (2, 3)
        if self.average == "micro":
            dims = (1, *dims)
        
        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, dims)
        
        # print("\nIntersection \t", intersection, intersection.shape)
        cardinality = torch.sum(input_soft + target_one_hot, dims)
        # print("Cardinality \t", cardinality, cardinality.shape)
        
        dice_score = 2.0 * intersection / (cardinality + self.eps)
        
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

#%%
class IoULoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 eps:float=1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.include_background=include_background

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
        
        input_soft = F.softmax(input, dim=1)
        
        # create the labels one hot tensor
        target_one_hot = tgm.losses.one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        
        #Ignore background?
        if not self.include_background:
            input_soft = input_soft[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        
        #Set diomension for the appropiate averaging
        dims: tuple[int, ...] = (2, 3)
        if self.average == "micro":
            dims = (1, *dims)
        
        # compute the actual iou score
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)
        union = cardinality - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = -iou + 1.0
        
        # reduce the loss across samples (and classes in case of `macro` averaging)
        iou_loss = torch.mean(iou_loss)#, axis=0)
        
        return iou_loss


#%%From https://github.com/anwai98/Loss-Functions/blob/main/loss-function-library-keras-pytorch.ipynb

class MyComboLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(MyComboLoss, self).__init__()
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
        
        # print("\nMyCombo", combo, combo.shape, dice, weighted_ce)
        
        return combo + 0.5 #As offset because betewwn [-0.5, 0.5]

#%%
class ComboLoss(nn.Module):
    def __init__(self, alpha, beta, average,include_background=False):
        super(ComboLoss, self).__init__()
        self.smooth=1
        self.alpha = alpha # < 0.5 penalises FP more, > 0.5 penalises FN more
        self.beta = beta #weighted contribution of modified CE loss compared to Dice loss
        self.eps = 1e-3
        self.average = average
        self.include_background=include_background
    
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
        # print(inputs.shape, targets.shape)
        
        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)
        #inputs_bool = (inputs_soft > 0.5).float()
        
        # create the labels one hot tensor
        target_one_hot = tgm.losses.one_hot(targets, num_classes=inputs.shape[1],
                                 device=inputs.device, dtype=inputs.dtype)
        
        # print(input_soft.shape, target_one_hot.shape)
        #Ignore background?
        if not self.include_background:
            input_soft = input_soft[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        # print(input_soft.shape, target_one_hot.shape)
        
        # compute the actual combo loss
        # dims = (1, 2, 3)
        #Set diomension for the appropiate averaging
        # set dimensions for the appropriate averaging
        dims: tuple[int, ...] = (2, 3)
        if self.average == "micro":
            dims = (1, *dims)
        
        #Dice
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice = torch.mean(dice)
        
        # print("\nDice", torch.min(dice), torch.max(dice), dice.shape)
        
        #BCE
        input_soft = torch.clamp(input_soft, self.eps, 1.0 - self.eps) #Avoid extreme cases ln(0) and ln (1)
        ce = -((self.beta * (target_one_hot * torch.log(input_soft))) +
                                ((1-self.beta) * ((1-target_one_hot) * torch.log(1 - input_soft))))
        nel = torch.numel(ce)
        ce = torch.sum(ce, dims)
        # print("CE", ce.shape, torch.min(ce), torch.max(ce))
        w_ce = ce/nel
        w_ce=torch.mean(w_ce)
        # print("BCE", torch.min(w_ce), torch.max(w_ce), w_ce.shape)
        
        #Combo
        # print(w_ce, dice)
        
        combo = (self.alpha * w_ce) - ((1 - self.alpha) * dice)
        # print("\nCombo",combo, combo.shape, dice, w_ce)
        return combo +0.5 #As offset because betewwn [-0.5, 0.5]

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

#%%NSD
#%%https://docs.monai.io/en/stable/metrics.html#surface-dice Using this implementation
#%%https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt
#%%
# def nsd(inputs:torch.Tensor, targets:torch.Tensor):
#     if not torch.is_tensor(inputs):
#         raise TypeError("Input type is not a torch.Tensor. Got {}"
#                         .format(type(inputs)))
#     if not len(inputs.shape) == 4:
#         raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                          .format(inputs.shape))
#     if not inputs.shape[-2:] == targets.shape[-2:]:
#         raise ValueError("input and target shapes must be the same. Got: {}"
#                          .format(inputs.shape, inputs.shape))
#     if not inputs.device == targets.device:
#         raise ValueError(
#             "input and target must be in the same device. Got: {}" .format(
#                 inputs.device, targets.device))
        
#     #Takes the ROI class
#     inputs_soft = F.softmax(inputs, dim=1)
#     sem_classes=['__background__', 'placenta']
#     sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
#     class_dim=1
#     inputs_bool = inputs_soft.argmax(class_dim) == sem_class_to_idx['placenta']
    
#     for inp, tar in inputs_bool, targets:
#         gt_boundary = find_boundaries(inp.cpu().numpy())
#         pr_boundary = find_boundaries(tar.cpu().numpy())
        
#%%Normalized surface distance 
# def nsdf(inputs, targets):
    
#     # compute softmax over the classes axis
#     inputs_soft = F.softmax(inputs, dim=1)
#     # # create the labels one hot tensor
#     targets_one_hot = tgm.losses.one_hot(targets, num_classes=inputs.shape[1],
#                               device=inputs.device, dtype=inputs.dtype)
    
#     #Convert predictions to binary mask
    
    
#     # Convert predictions binary mask in format BCHW
#     sem_classes=['background', 'roi1']
#     sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
    
#     class_dim=1
#     inputs_bool0 = inputs_soft.argmax(class_dim) == sem_class_to_idx['background']
#     inputs_bool1 = inputs_soft.argmax(class_dim) == sem_class_to_idx['roi1']
#     inputs_bool = torch.cat((inputs_bool0, inputs_bool1)).unsqueeze(0)
    
#     # Convert targets binary mask in format BCHW
#     targets_bool0 = targets_one_hot.argmax(class_dim) == sem_class_to_idx['background']
#     targets_bool1 = targets_one_hot.argmax(class_dim) == sem_class_to_idx['roi1']
#     targets_bool = torch.cat((targets_bool0, targets_bool1)).unsqueeze(0)

#     # print(inputs_soft.shape, targets_one_hot.shape)
    
#     # inputs_soft = 
#     # yoh 
    
#     # print('inputs', 
#     #       torch.max(inputs_bool).detach().item(),
#     #       torch.min(inputs_bool).detach().item(
#     #           ))
    
#     # print('targets',
#     #       torch.max(targets_bool).detach().item(),
#     #       torch.min(targets_bool).detach().item())
    
#     nsd=monai.metrics.compute_surface_dice(inputs_bool, targets_bool, [2.0], use_subvoxels=False )
#     # print(nsd, nsd.shape, torch.max(nsd), torch.min(nsd))
    
#     return nsd.detach().item()