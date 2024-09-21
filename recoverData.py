# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:57:35 2023

@author: josef
"""

#%% Import libraries
from pymoo.mcdm.pseudo_weights import PseudoWeights

import os
import torch
import pandas as pd
import numpy as np
import random
import pickle
from torchinfo import summary
# import torch.multiprocessing as mp
import torch.optim as optim
from deap import tools
from deap import creator, base
import deap.gp as gp
import time
import operator
import matplotlib.pyplot as plt

import gp_restrict
from strongGPDataType import (moduleTorch, moduleTorchL, moduleTorchSe, moduleTorchCn, moduleTorchCt, moduleTorchP,
                              outChConv, outChSConv, kernelSizeConv, dilationRate,
                              tetha,wArithm)
from operators.functionSet import (convolution, sep_convolution,
                             res_connection, dense_connection,
                             se,
                             add, sub, cat,
                             maxpool, avgpool)
from toolbox_functions import (make_model, evaluationMP, evaluation, evaluationMO,
                               save_ind, save_graphtv, save_graphtvd,
                               identifier)

from model.loss_f import ComboLoss
from model.train_valid import train_and_validate
from model.predict import test
from data.loader import loaders
# from algorithm import NASGP_Net
# from algorithm_NASGPNet import eaNASGPNet
from algorithm_MONASGPNet import eaMONASGPNet, dominate_relation, eaMONASGPNet_CHKP
from utils.save_utils import saveEvolutionaryDetails, saveTrainingDetails, save_execution
from utils.deap_utils import statics_, save_statics, show_statics, functionAnalysis

import data.dataSplit as dataSplit
import data.dataStatic as dataStatic

import pickle

#%%Pset
pset = gp.PrimitiveSetTyped("main", [moduleTorch], moduleTorchP)

#Pooling Layer
pset.addPrimitive(maxpool, [moduleTorchL], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchL], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchCn], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchCn], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchSe], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchSe], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchCt], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchCt], 
                  moduleTorchP, name='apool')


#Feature Construction Layer Optional
#L,L; L,Cn; Cn,L; Cn,Cn; 
pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchL, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchL, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchCn], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchCn, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchCn, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchCn], 
                  moduleTorchCt, name='cat')

#Se,Se; Se,Cn; Cn,Se;
pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchSe], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchSe,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchSe,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchSe], 
                  moduleTorchCt, name='cat')

#Cn,Se; Se,Cn;
pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchSe], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchCn], 
                  moduleTorchCt, name='cat')

# Recalibarion Layer Optional: Squeeze and Excitation Operation
pset.addPrimitive(se, [moduleTorchL],
                  moduleTorchSe, name='se')
pset.addPrimitive(se, [moduleTorchCn],
                  moduleTorchSe, name='se')

# #Feature Connection Layer Optional
pset.addPrimitive(dense_connection, [moduleTorchL, tetha],#, kernelSizeConv, kernelSizeConv
                  moduleTorchCn, name='dCon')
pset.addPrimitive(res_connection, [moduleTorchL],
                  moduleTorchCn, name='rCon')
   
#Feature Extraction Layer
pset.addPrimitive(convolution, [moduleTorch, outChConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorch, name='conv')
pset.addPrimitive(sep_convolution, [moduleTorch, outChSConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorch, name='sconv')

pset.addPrimitive(convolution, [moduleTorch, outChConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorchL, name='conv')
pset.addPrimitive(sep_convolution, [moduleTorch, outChSConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorchL, name='sconv')

#Terminals
pset.addEphemeralConstant('outChConv', lambda:outChConv() , outChConv)
pset.addEphemeralConstant('outChSConv', lambda:outChSConv() , outChSConv)
pset.addEphemeralConstant('ksConv', lambda:kernelSizeConv() , kernelSizeConv)
pset.addEphemeralConstant('dilationR', lambda:dilationRate() , dilationRate)
pset.addEphemeralConstant('tetha', lambda:tetha(), tetha)
pset.addEphemeralConstant('w', lambda:wArithm() , wArithm)

pset.renameArguments(ARG0="mod")

#%%Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
                # dices=list, ious=list, hds=list,
                # train_loss=list, valid_loss=list,
                params=int, dice=float,
                location=None)

#%% Data
# path_images='/home/202201016n/serverBUAP/datasets/images_PROMISE'
path_images = 'C:/Users/josef/serverBUAP/datasets/images_DRIVE'
in_channels = 3

"""Get train, valid, test set and loaders"""
train_set, valid_set, test_set = dataSplit.get_data(0.7, 0.15, 0.15, path_images)
# train_set, valid_set, test_set = dataStatic.get_data(path_images)

IMAGE_HEIGHT = 256#288 
IMAGE_WIDTH = 256#480
NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders
BATCH_SIZE = 1

dloaders = loaders(train_set, valid_set, test_set, batch_size=BATCH_SIZE, 
                 image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, in_channels=in_channels,
                 num_workers=NUM_WORKERS)

ruta = "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/EEs/MO-1/Proyecto/first_attempt/MONASGPNet1/recover/"
if not os.path.exists(ruta):
    os.makedirs(ruta)

#%%Recover data of eack .pkl
ruta1_pkl = "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/EEs/MO-1/Proyecto/first_attempt/MONASGPNet1/MONASGPNet1.pkl"
ruta2_pkl = "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/EEs/MO-1/Proyecto/first_attempt/MONASGPNet1/checkpoint.pkl"

#Recover pickle1
with open(ruta1_pkl, "rb") as cp_file:
    cp = pickle.load(cp_file)
# print(cp.keys())
    sz_e    = cp["SZ_E"]
    sz_m = cp["SZ_M"]
    mnr_p = cp["MNR_p"]
    sz_up = cp["SZ_UP"]
    its_uga = cp["ITS_uGA"]
    div_agrid = cp["DIV_AGRID"]
    cxpb = cp["CXPB"]
    mutpb = cp["MUTPB"]
    rep_cycle = cp["REP_CYCLE"]
    Ef = cp["E"]
    
#Recover pickle2
with open(ruta2_pkl, "rb") as cp_file:
    cp2 = pickle.load(cp_file)
    
    it_start = cp2["it"]
    E        = cp2["E"]
    M        = cp2["M"]
    cache    = cp2["cache"]
    g_log    = cp2["g_log"]
    
    no_evs   = cp2["no_evs"]
    delta_time     = cp2["time"]
    t_frame  = cp2["t_frame"]
    # random.setstate(cp["rndstate"])

    
#%%Tasa marginal de sustituci[on]
def distancia(x,y):

    return np.sqrt(np.sum(((x - y) / 1) ** 2))

reference_point = np.array([0,0])
dists=[]
for ind in E:
    d = distancia(np.array(ind.fitness.values), reference_point)
    dists.append(d)

idx=np.argmin(dists)
sel = E[idx]

#%%
from pymoo.mcdm.pseudo_weights import PseudoWeights
Ef = np.array(Ef)

approx_ideal = Ef.min(axis=0)
approx_nadir = Ef.max(axis=0)

nF = (Ef - approx_ideal) / (approx_nadir - approx_ideal)

weights = np.array([0.9, 0.1])
i, wi = PseudoWeights(weights).do(Ef, return_pseudo_weights=True)

selected = E[i]

#%%Train again the best architecture for reliable compatarion with U-Net
#Training_parameters
nepochs = 1
alpha = 0.5
beta = 0.4
lossfn = ComboLoss(alpha=alpha, beta=beta)
lr = 0.0001
training_parameters = {'num_epochs':nepochs, 'loss_f':lossfn, 'learning_rate':lr}

model=make_model(selected, in_channels=3, pset=pset)

"""Train, val and test loaders"""
train_loader, _= dloaders.get_train_loader()
val_loader, _  = dloaders.get_val_loader()
test_loader, _ = dloaders.get_test_loader()

"""Optimizer"""
optimizer = optim.Adam(model.parameters(), lr=lr)

"""Device"""
device='cuda:0'

"""For save images or model"""
save_model_flag=True
save_images_flag=True
load_model=False

#%%Train, validate and test
"""Train and valid model"""
train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
    model, train_loader, val_loader,
    nepochs, optimizer, lossfn,
    device, load_model, save_model_flag,
    ruta=ruta, verbose=True
    )

"""Test model"""
dices, ious, hds = test(test_loader, model, lossfn,
                        save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)

#%%Attributes assigned
# best_model=toolbox.clone(best)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# selected.fitness.values = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params),
selected.dice = np.mean(dices)
selected.params = params

selected.train_loss = train_loss
selected.valid_loss = valid_loss
selected.train_dice = train_dice
selected.valid_dice = valid_dice

selected.dices = dices
selected.ious = ious
selected.hds = hds

#%%NNo. parameters
"""No of parameters"""
model = model.to(device)
model_stats=summary(model, (1, in_channels, 256, 256), verbose=0)
summary_model = str(model_stats)

#%%Retrain Details
saveTrainingDetails(training_parameters, dloaders, 
                    selected, summary_model,
                    filename=ruta+'/Retrain_best_details.txt')

#Recover pickle2
# with open(ruta2_pkl, "rb") as cp_file:
#     cp = pickle.load(cp_file)
# # print(cp.keys())
# start_gen    = cp["generation"]
# population   = cp["population"]
# offspring    = cp["offspring"]
# invalid_ind  = cp["invalid_ind"]
# idx          = cp["idx"]
# elitism_inds = cp["elitism_inds"]

# no_evs       = cp["no_evs"]
# delta_t      = cp["delta_t"]
# cache        = cp["cache"]

# halloffame   = cp["halloffame"]
# logbook      = cp["logbook"]
# random.setstate(cp["rndstate"])



