# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 00:38:15 2022

@author: josef
"""

import torch.nn as nn
from operators.moduleConv import moduleconv
import copy
# from dwsBlock import LazySeparableConv2d

class ResBlockLayers(nn.Module):
    def __init__(self, moduleList):
        super(ResBlockLayers, self).__init__()
        #modules
        self.moduleList=moduleList
        self.identity_downsample=None
        self.finalReLU=nn.ReLU()
        
        #change last layer
        lastLayer=moduleList[-1]
        if hasattr(lastLayer.conv, 'depthwise'):
            lastLayer.make_layer(type='separableRes')
        else:
            lastLayer.make_layer(type='regularRes')
        
        #add identity downsampling
        if moduleList[-1].out_channels!=moduleList[0].in_channels:
            self.identity_downsample=moduleconv(moduleList[0].in_channels, moduleList[-1].out_channels, 
                                                (1,1), 1, moduleList[-1].groupsGN, 'regularRes')
        
        #attributes
        self.in_channels=self.moduleList[0].in_channels
        self.out_channels=self.moduleList[-1].out_channels
            
    def forward(self, x):
        identity = x.clone()
        
        for layer in self.moduleList:
            x = layer(x)
        
        if self.identity_downsample!=None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.finalReLU(x)
        return x

class ResBlockConnection(nn.Module):
    def __init__(self, moduleList, bseq):
        super(ResBlockConnection, self).__init__()
        #sequential modules
        self.moduleList=list(moduleList)
        #connect list
        self.make_connection(copy.deepcopy(self.moduleList), bseq)
    
    def list2listnn(self, moduleList):
        return nn.ModuleList(moduleList)
        
    def make_connection(self, moduleList, bseq=2):
        #modules
        moduleList=self.list2listnn(moduleList) #to convert
        self.moduleListRes=nn.ModuleList([])
        
        
        #groups every bseq layers(2) except the first
        SeqLayers=nn.ModuleList()
        if len(moduleList)%bseq==1 and len(moduleList)>bseq:
            self.moduleListRes.append(moduleList[0])
            start=1
        else:
            start=0
            
        if len(moduleList)>bseq:
            for idx in range(start, len(moduleList)):
                if len(SeqLayers)%bseq<bseq:
                    SeqLayers.append(moduleList[idx])
                if len(SeqLayers)%bseq==0:
                    self.moduleListRes.append(ResBlockLayers(SeqLayers))
                    SeqLayers=nn.ModuleList()
        elif len(moduleList)<=bseq and start!=1:
            self.moduleListRes.append(ResBlockLayers(moduleList))
        
        #attributes
        self.out_channels=self.moduleListRes[-1].out_channels
    
    def forward(self, x):
        for layer in self.moduleListRes:
            x = layer(x)
        
        return x
        
def resConnection(moduleList, bseq):
    resList=ResBlockConnection(moduleList, bseq)
    return resList