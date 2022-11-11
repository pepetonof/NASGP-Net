# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:18:15 2022

@author: josef
"""

import torch.nn as nn

#%%Depth-wise separable convolution. 
#Inspired from: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
class LazySeparableConv2d(nn.Module):    
    def __init__(self, out_channels, kernel_size1, kernel_size2, dilation_rate=1, groupsGNdepth=1, groupsGNpoint=1, bias=False):
        super(LazySeparableConv2d, self).__init__()
        dummy=4
        self.depthwise = nn.Sequential(nn.LazyConv2d(dummy, (kernel_size1, kernel_size2),
                                       stride=1, padding='same', dilation=dilation_rate, 
                                       groups=1, bias=False),
                                       nn.GroupNorm(groupsGNdepth, dummy),
                                       nn.ReLU(inplace=True)
                                       )
        self.pointwise = nn.Sequential(nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
                                       nn.GroupNorm(groupsGNpoint, out_channels),
                                       nn.ReLU(inplace=True)
                                       )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class SeparableConv2d(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2, dilation_rate=1, groupsGN=1, bias=False):
        super(SeparableConv2d, self).__init__()
        dummy=4
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, dummy, (kernel_size1, kernel_size2),
                                       stride=1, padding='same', dilation=dilation_rate, 
                                       groups=1, bias=False),
                                       nn.GroupNorm(groupsGN, dummy),
                                       nn.ReLU(inplace=True)
                                       )
        self.pointwise = nn.Sequential(nn.Conv2d(dummy, out_channels, kernel_size=1, bias=False),
                                       nn.GroupNorm(groupsGN, out_channels),
                                       nn.ReLU(inplace=True)
                                       )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class SeparableConv2dRes(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2, dilation_rate=1, groupsGN=2, bias=False):
        super(SeparableConv2dRes, self).__init__()
        dummy=4
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, dummy, (kernel_size1, kernel_size2),
                                        stride=1, padding='same', dilation=dilation_rate, 
                                        groups=1, bias=False),
                                        nn.GroupNorm(groupsGN, dummy),
                                        nn.ReLU(inplace=True)
                                        )
        self.pointwiseRes = nn.Sequential(nn.Conv2d(dummy, out_channels, kernel_size=1, bias=False),
                                        nn.GroupNorm(groupsGN, out_channels),
                                        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwiseRes(out)
        return out