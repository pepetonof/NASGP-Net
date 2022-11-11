# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:04:13 2022

@author: josef
"""
import torch.nn as nn
from operators.dwsBlock import SeparableConv2d, SeparableConv2dRes

#%%For ResBlocks
class Conv2dRes(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2, dilation_rate=1, groupsGN=2, bias=False):
        super(Conv2dRes, self).__init__()
        self.convRes = nn.Sequential(nn.Conv2d(in_channels, out_channels, (kernel_size1, kernel_size2), 
                                      stride=1, padding='same', dilation=dilation_rate, 
                                      groups=1, bias=False),
                                      nn.GroupNorm(groupsGN,out_channels),
                                      )

    def forward(self, x):
        out = self.convRes(x)
        return out

#%%moduleconv object
class moduleconv(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, dilation=None, groupGN=4, type='regular'):
        super(moduleconv, self).__init__()
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.out_channels=out_channels
        self.dilation=dilation
        self.groupsGN=groupGN
        
        self.type=type
        self.make_layer(self.type)
    
    def make_layer(self, type='regular'):
        if type=='regular':
            self.conv=nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_size[0], self.kernel_size[1]), 
                          1, 'same', dilation=self.dilation, bias=False),
                nn.GroupNorm(self.groupsGN, self.out_channels),
                nn.ReLU(inplace=True)
                )
            
        elif type=='regularRes':
            self.conv=Conv2dRes(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1], 
                                self.dilation, groupsGN=self.groupsGN)
            
        elif type=='separable':
            self.conv=SeparableConv2d(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1], 
                                      self.dilation, groupsGN=self.groupsGN)
        
        elif type=='separableRes':
            self.conv=SeparableConv2dRes(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1], 
                                         self.dilation, groupsGN=self.groupsGN)
           
    
    def forward(self, x):
        x = self.conv(x)
        return x