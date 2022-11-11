# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 01:45:59 2022

@author: josef

inspired from: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
"""
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        if channel<=reduction:
            reduction=channel//2
        self.out_channels=channel
        self.reduction=reduction
        self.make_layer(self.out_channels, reduction)
        
    def make_layer(self, channel, reduction):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out=x * y.expand_as(x)
        return out
    
class SEConv(nn.Module):
    def __init__(self, moduleconv, reduction):
        super(SEConv, self).__init__()
        self.conv=moduleconv
        self.out_channels=moduleconv.out_channels
        self.reduction=reduction

        # channel=moduleconv.out_channels
        if self.out_channels//self.reduction==0:
            self.reduction = self.out_channels//2
        self.make_layer(self.out_channels, self.reduction)
        # self.se=SELayer(channel, reduction)
    
    def make_layer(self, channel, reduction):
        self.se=SELayer(channel, reduction)
        
    def forward(self, x):
        out=self.conv(x)
        out=self.se(out)
        
        return out

def seBlock(moduleList, reduction):
    channel=moduleList[-1].out_channels
    se=SELayer(channel, reduction)
    moduleList.append(se)
    
    return moduleList
        
