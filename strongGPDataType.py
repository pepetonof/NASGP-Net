# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 05:08:59 2021

@author: josef
"""
#%%Packages
import torch.nn as nn
import random
from torch.nn import Module

#%%For function set
class moduleTorch:
    def __init__(Module):
        pass
    
class moduleTorchL:
    def __init__(Module):
        pass

class moduleTorchSe:
    def __init__(Module):
        pass

class moduleTorchCn:
    def __init__(Module):
        pass  

class moduleTorchCt:
    def __init__(Module):
        pass

class moduleTorchP:
    def __init__(Module):
        pass

    
#%%For terminal set
#Convolution and separable convolution
class outChConv(int):
    def __new__(cls):
        value = random.choice([8,16,32])
        # value = random.randint(8,16)
        # value = random.choice([8,12,16,24,32])
        return  value

class outChSConv(int):
    def __new__(cls):
        value = random.choice([8,16,32])
        # value = random.randint(8,32)
        # value = random.choice([8,12,16,24,32])
        return  value
    
class kernelSizeConv:
    def __new__(cls):
        value = random.choice([3,5,7])
        return  value

class dilationRate:
    def __new__(cls):
        value = random.choice([1,2])
        return  value

# class reduction:
#     def __new__(cls):
#         value = random.choice([8,16])
#         return  value

# #ResBlock and 
# class outChRes:
#     def __new__(cls):
#         value = random.choice([8,16,32,64])
#         # value = random.randint(8,32)
#         # value = random.choice([8,12,16,24,32])
#         return value

# class numModSeqRes:
#     def __new__(cls):
#         value = random.choice([2,3])
#         return value
    
# class numBlocksRes:
#     def __new__(cls):
#         value = random.choice([1,2])
#         return value

# #DenseBlock
# class growthRate:
#     def __new__(cls):
#         value = random.randint(8,12)#4 a 12? 8 a 24?
#         return value
# class L:
#     def __new__(cls):
#         value = random.randint(5,10)
#         return value
    
class tetha:
    def __new__(cls):
        epsilon=3E-1
        value = round(random.uniform(epsilon,0.8),1)
        return value

# #Pooling
# class kernelSizePool:
#     def __new__(cls):
#         value = random.choice([2,4])
#         return  value

#Arithmetic
class wArithm:
    def __new__(cls):
        epsilon=1E-1
        value = round(random.uniform(epsilon,1),2)
        # value = round(random.random(),2)
        return  value