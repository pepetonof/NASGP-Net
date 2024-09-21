# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 05:07:38 2022

@author: josef
"""

import torch
import torch.nn as nn
# import torchvision.transforms.functionalJ as TF
import torchvision.transforms.functional as TF
import copy

from operators.moduleConv import moduleconv#, containerseq
from operators.seBlock import seBlock
from operators.ResConnection import resConnection
from operators.DenseConnection import denseConnection

#%%convolution
def convolution(containerseq, out_channels, kernel_size1, kernel_size2, dilation_rate):
    # print(containerseq)
    groupsGN = 4
    #if len(containerseq)==2:
    if type(containerseq)==list:
        in_channels=containerseq[1]
        containerseq=containerseq[0]
    else:
        in_channels=containerseq[-1].out_channels
    
    # moduleConv=nn.Sequential(
    #     nn.Conv2d(in_channels, out_channels, (kernel_size1, kernel_size2), 1, 'same', dilation=dilation_rate, bias=False),
    #     nn.GroupNorm(groupsGN, out_channels),
    #     nn.ReLU(inplace=True)
    #     )
    # module=moduleconv(moduleConv, in_channels, out_channels, (kernel_size1, kernel_size2), dilation=dilation_rate)
    module=moduleconv(in_channels, out_channels, (kernel_size1, kernel_size2), dilation_rate, groupsGN, 'regular')
    
    containerseqc=copy.deepcopy(containerseq)
    containerseqc.append(module)
    del containerseq
    
    return containerseqc

#%%sep convolution
def sep_convolution(containerseq, out_channels, kernel_size1, kernel_size2, dilation_rate):
    # print(containerseq)
    groupsGN = 4
    # if len(containerseq)==0:
    if type(containerseq)==list:
        in_channels=containerseq[1]
        containerseq=containerseq[0]
        #in_channels=3
    else:
        in_channels=containerseq[-1].out_channels

    # moduleSepConv=SeparableConv2d(in_channels, out_channels, kernel_size1, kernel_size2, dilation_rate=dilation_rate, groupsGN=groupsGN)
    # module=moduleconv(moduleSepConv, in_channels, out_channels, (kernel_size1, kernel_size2), dilation=dilation_rate)
    module=moduleconv(in_channels, out_channels, (kernel_size1, kernel_size2), dilation_rate, groupsGN, 'separable')
    # del moduleSepConv
    
    containerseqc=copy.deepcopy(containerseq)
    containerseqc.append(module)
    del containerseq

    return containerseqc

#%%Squeeze and Excitation operation
def se(containerseq):
    # print(containerseq)
    containerseq = seBlock(containerseq, 8)
    return containerseq

#%%DenseConnection and ResConnections
def res_connection(containerseq):
    moduleResConnection=resConnection(containerseq, 2)
    return nn.ModuleList([moduleResConnection])

def dense_connection(containerseq, tetha):
    moduleDenseConnection=denseConnection(containerseq, tetha)
    return nn.ModuleList([moduleDenseConnection])

#%%Arithmetic Layer
class MyModule_AddSubCat(nn.Module):
    def __init__(self, moduleList, flag, n1, n2):
        super(MyModule_AddSubCat, self).__init__()            
        self.moduleList=nn.ModuleList(moduleList)
        
        self.flag=flag
        self.n1=n1
        self.n2=n2
        
        if self.flag==2:
            self.out_channels=self.moduleList[0][-1].out_channels + self.moduleList[1][-1].out_channels   
        else:
            self.out_channels=max(self.moduleList[0][-1].out_channels, self.moduleList[1][-1].out_channels)
   
    def forward(self, x):
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        # x=x.to(device=device)
        outputs=[]
        #print('ForwardAdd')
        for idx, m in enumerate(self.moduleList):
            xclone=x.clone()
            for e in m:
                xclone=e(xclone)
                # print('AddSubCat', idx, xclone.shape)
            outputs.append(xclone)
            # del x
            
            # #forward for each element in list
            # outputs.append(m(x))
            
            #print('AddM', outputs[idx].shape)
            #Reshaping for add or sub
            if outputs[idx].shape != outputs[idx-1].shape and idx>0:                
                #Reshape image H, W
                outputs[idx]=TF.resize(outputs[idx], size=outputs[idx-1].shape[2:])
                #print('AddResh', outputs[idx].shape)
                
                #Padd channels (C) if is necessesary over the tensor with fewer channels
                if outputs[idx].shape[1]>outputs[idx-1].shape[1] and (self.flag==0 or self.flag==1):
                    pad_ch=outputs[idx].shape[1]-outputs[idx-1].shape[1]
                    zeros = torch.zeros(outputs[idx].shape[0], pad_ch, 
                                        outputs[idx].shape[2], outputs[idx].shape[3])
                    
                    #print('outbef, zeros', outputs[idx-1].device, zeros.device)
                    device=outputs[idx-1].device
                    zeros=zeros.to(device=device)
                    #print('outaft, zeros', outputs[idx-1].device, zeros.device)
                    
                    outputs[idx-1]= torch.cat((outputs[idx-1], zeros), 1)
                    # print('AddCh',outputs[idx].shape)
                    
                elif outputs[idx].shape[1]<outputs[idx-1].shape[1] and (self.flag==0 or self.flag==1):
                    pad_ch=outputs[idx-1].shape[1]-outputs[idx].shape[1]
                    zeros = torch.zeros(outputs[idx-1].shape[0], pad_ch, 
                                        outputs[idx-1].shape[2], outputs[idx-1].shape[3])
                    
                    #print('outbef, zeros', outputs[idx].device, zeros.device)
                    device=outputs[idx].device
                    zeros=zeros.to(device=device)
                    #print('outaft, zeros', outputs[idx].device, zeros.device)
                    
                    outputs[idx]= torch.cat((outputs[idx], zeros), 1)                        
                    # print('AddCh',outputs[idx].shape)
                    
        if self.flag==0:
            return outputs[0]*self.n1 + outputs[1]*self.n2
        elif self.flag==1:
            return outputs[0]*self.n1-outputs[1]*self.n2
        elif self.flag==2:
            return torch.cat((outputs[0], outputs[1]), dim=1)

def add(module1, n1, module2, n2):
    module=MyModule_AddSubCat([module1, module2], 0, n1, n2) 
    
    return nn.ModuleList([module])

def sub(module1, n1, module2, n2):
    module=MyModule_AddSubCat([module1, module2], 1, n1, n2)
    
    return nn.ModuleList([module])

def cat(module1, module2):
    module=MyModule_AddSubCat([module1, module2], 2, 0, 0)
    
    return nn.ModuleList([module])


#%%Pooling Layer
class CNNCell(nn.Module):
    def __init__(self, moduleList):
        super(CNNCell, self).__init__()
        self.moduleListCell=nn.ModuleList(moduleList)
        self.out_channels=moduleList[-2].out_channels
        # print(self.out_channels)
    
    def forwardpart(self, x, part):
        if part=='first':
            #x=self.moduleListPool[0](x)
            for l in self.moduleListCell[:-1]:
                # print(x.shape)
                x=l(x)
        elif part=='second':
            # x=self.moduleListPool[1](x)
            x=self.moduleListCell[-1](x)
        else:
            raise ValueError("Invalid part")
        return x
    
    def forward(self, x):
        # print(x.shape)
        for l in self.moduleListCell:
            # print(l)
            x=l(x)
            # print(x.shape)
        return x

def maxpool(module):
    modulePool=nn.MaxPool2d(2, 2)
    module.append(modulePool)
    cell=CNNCell(module)
        
    return cell

def avgpool(module):
    modulePool=nn.AvgPool2d(2, 2)
    module.append(modulePool)
    cell=CNNCell(module)
    
    return cell

