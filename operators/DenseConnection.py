# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:42:54 2022

@author: josef
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F
import copy

from operators.moduleConv import moduleconv


def bn_function_factory(conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        output = conv(concated_features)
        return output

    return bn_function


class DenseLayerConnection(nn.Module):
    def __init__(self, moduleConv, in_channels_mod, drop_rate=0.2):
        super(DenseLayerConnection, self).__init__()
        #modules
        self.moduleConv=moduleConv
        #module change in channels
        self.moduleConv.in_channels=in_channels_mod
        self.moduleConv.make_layer(type=moduleConv.type)
        
        #attributes
        self.in_channels=in_channels_mod
        self.out_channels=self.moduleConv.out_channels
        self.drop_rate=drop_rate
        
    def forward(self, *prev_features):
        bn_function = bn_function_factory(self.moduleConv)
        output = bn_function(*prev_features)
        
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)
        return output


class DenseBlockConnection(nn.Module):
    def __init__(self, moduleList, tetha, drop_rate=0.2):
        super(DenseBlockConnection, self).__init__()
        self.tetha=tetha
        #sequential modules
        self.moduleList=list(moduleList)
        # print('Dense 1', self.moduleList)
        self.make_connection(copy.deepcopy(self.moduleList), self.tetha)
        # print('Dense 2', self.moduleList)
     
    def list2listnn(self, moduleList):
        return nn.ModuleList(moduleList)
     
    def make_connection(self, moduleList, tetha):
        #module for Dense Connection
        moduleList=self.list2listnn(moduleList)
        in_channels=moduleList[0].in_channels
        self.moduleListDense=nn.ModuleList([moduleList[0]])
        
        for layer in moduleList[1:]:
            # print('make layer', layer.in_channels, in_channels)
            in_channels=layer.in_channels+in_channels
            self.moduleListDense.append(DenseLayerConnection(layer, in_channels))
        
        #module for transition layer
        groupsGN=moduleList[-1].groupsGN
        compression=int(tetha*(in_channels+self.moduleListDense[-1].out_channels))
        g=self.groupsTransition(compression, groupsGN)
        self.transitionLayer=moduleconv(in_channels+self.moduleListDense[-1].out_channels, compression, 
                                        (1,1), 1, g, 'regular')
        
        #attributes
        self.out_channels=compression
        
    def groupsTransition(self, compression, init_groups):
        div=[]
        if compression%init_groups==0:
            return init_groups
        else:
            for i in range(2, compression+1):
                if compression%i==0:
                    div.append(i)
            return min(div) 
        
    def forward(self, init_features):
        features = [init_features]
        
        for layer in self.moduleListDense:
            new_features = layer(*features)
            features.append(new_features)
        outBlock=torch.cat(features, 1)

        outBlock=self.transitionLayer(outBlock)
        
        return outBlock

def denseConnection(moduleList, tetha):
    denseList=DenseBlockConnection(moduleList, tetha)
    # print(moduleList)
    return denseList