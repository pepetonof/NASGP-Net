# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:42:23 2022

@author: josef

inspired from: https://towardsdatascience.com/paper-review-densenet-densely-connected-convolutional-networks-acf9065dfefb

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from dwsBlock import LazySeparableConv2d

def bn_function_factory2(dws_conv):

    def bn_function(*inputs):
        
        concated_features = torch.cat(inputs, 1)
        output = dws_conv(concated_features)
        return output

    return bn_function

def bn_function_factory(norm, relu, conv):
    
    def bn_function(*inputs):
        
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    
    def __init__(self, growth_rate, drop_rate=0): #bn_size
        
        super(DenseLayer, self).__init__()
        # self.add_module('norm1', nn.LazyBatchNorm2d()),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.LazyConv2d(bn_size * growth_rate,
        #                 kernel_size=1, stride=1, bias=False)),
        
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        # #                 kernel_size=(h, w), stride=1, padding='same', bias=False)),
        # self.add_module('conv2', nn.LazyConv2d(growth_rate,
        #                 kernel_size=3, stride=1, padding='same', bias=False)),  
        
        self.dws_conv=LazySeparableConv2d(growth_rate, 3, 3, 1)
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        
        bn_function = bn_function_factory2(self.dws_conv)
        output = bn_function(*prev_features)
        # bottleneck_output = bn_function(*prev_features)
        # new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            
            output = F.dropout(output, p=self.drop_rate, training=self.training)
        
        return output#new_features
    
class DenseBlock(nn.Module):

    def __init__(self, num_layers, tetha , growth_rate,  drop_rate):#  bn_size,   ..., height, width,
    
        super(DenseBlock, self).__init__()
        self.moduleListDense = nn.ModuleList()
        self.tetha=tetha
        for i in range(num_layers):       
            
            layer = DenseLayer(
                # num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                # bn_size=bn_size,
                # h=height,
                # w=width,
                drop_rate=drop_rate
            )
            
            self.moduleListDense.append(layer)
            
            # self.add_module('denselayer%d' % (i + 1), layer)
        self.flag=False
    def forward(self, init_features):
    
        features = [init_features]
        
        for layer in self.moduleListDense:
            new_features = layer(*features)
            features.append(new_features)
            
        # for name, layer in self.named_children():
        #     new_features = layer(*features)
        #     features.append(new_features)
        
        outBlock=torch.cat(features, 1)
        
        if self.flag==False:
            compression=int(self.tetha*outBlock.shape[1])
            self.transitionLayer= nn.Sequential(
                nn.Conv2d(
                    outBlock.shape[1],
                    compression,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm2d(compression),
                nn.ReLU(inplace=True),
            )
            self.flag=True
        outBlock=self.transitionLayer(outBlock)
        
        return outBlock#outTransition#torch.cat(features, 1)
    
def denseBlock(growth_rate, L, tetha):#, h, w
    return DenseBlock(num_layers=L, tetha=tetha, growth_rate=growth_rate, drop_rate=0.2)
    # return DenseBlock(num_layers=L, bn_size=bn_size, growth_rate=growth_rate, drop_rate=0.5)#height=h, width=w
    
    
# denseBlock1=denseBlock(growth_rate=32, tetha=0.5, L=4)
# x=torch.rand(1,3,295,480)
# out=denseBlock1(x)