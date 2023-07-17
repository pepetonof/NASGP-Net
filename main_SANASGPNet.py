# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:20:14 2023

@author: josef
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:52:10 2023

@author: josef
"""

# import sys
# sys.path.append( 'C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/NASGP-Net' )


#%% Import libraries
import os
import torch
import pandas as pd
import numpy as np
import random
from torchinfo import summary
import torch.multiprocessing as mp
import torch.optim as optim
from deap import tools
from deap import creator, base
import deap.gp as gp
import time
import operator

import gp_restrict
from strongGPDataType import (moduleTorch, moduleTorchL, moduleTorchSe, moduleTorchCn, moduleTorchCt, moduleTorchP,
                              outChConv, outChSConv, kernelSizeConv, dilationRate,
                              tetha,wArithm)
from operators.functionSet import (convolution, sep_convolution,
                             res_connection, dense_connection,
                             se,
                             add, sub, cat,
                             maxpool, avgpool)
from toolbox_functions import (make_model, evaluation, 
                               save_ind, identifier)

from objective_functions import evaluate_Segmentation, evaluate_NoParameters

from model.loss_f import ComboLoss
from model.train_valid import train_and_validate
from model.predict import test
from data.loader import loaders
# from algorithm import NASGP_Nets
# from surrogate_algorithm import NASGP_Net
# from surrogate_algorithm2 import eaNASGP_Net
# from surrogate_algorithm3 import eaNASGP_Net
from algorithm_SANASGPNet import eaNASGP_Net, eaNASGP_Net2, assign_attributes
from utils.save_utils import saveEvolutionaryDetails, saveTrainingDetails, save_execution
from utils.deap_utils import statics_, save_statics, show_statics, functionAnalysis

import data.dataSplit as dataSplit
import data.dataStatic as dataStatic

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
import torch.nn as nn
from operators.DenseConnection import DenseBlockConnection
from operators.functionSet import MyModule_AddSubCat

from tqdm import tqdm

# from surrogate_NASGPNet import assign_attributes

import itertools
import collections

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
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, 
                             params=int, dice=float,
                             dice_p1=float, dice_p2=float, dice_m=float)

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("selectElitism", tools.selBest)

toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)

toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate_uniform", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.register("mutate_shrink", gp.mutShrink)
toolbox.register("mutate_replace", gp.mutNodeReplacement, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate_eph", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate_shrink", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate_replace", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate_uniform", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

toolbox.register("save_ind", save_ind)
# toolbox.register("plt_ind", plt_ind)

toolbox.register("identifier", identifier, length=10)

#%%%Surrogate model using Random Forest Regressor
def count_functions(ind):
    def count(ind):
        dic={}
        string=str(ind)
        specialchar = '¿?¡!{}[]()<>\'""+-*/.:,;~…‘’“”``´´^¨#$%&_—°|¬1234567890«»×=//\\'
        ignore=set("mod".split())
        for c in specialchar:
            string=string.replace(c," ")
        string_lst=string.split()
        string_lst=list(word for word in string_lst if word not in ignore)
        for w in string_lst:
            if w in dic:
                dic[w]+=1
            else:
                dic[w]=1
        return dic
        
    count=count(ind)
    funciones=list(pset.context.keys())[1:]
    faltantes=[f for f in funciones if f not in set(count.keys())]
    
    if len(faltantes)>0:
        for f in faltantes:
            count[f]=0

    count=dict(sorted(count.items()))
    
    return count

def count_LayersAndFilters(model):
    filters=0
    conv_layers=0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.ConvTranspose2d):
            conv_layers+=1
            filters+=layer.out_channels
    
    return conv_layers, filters

def sum_terminals(model):
    #Sum of kernel size (height and width)
    kx=0;ky=0
    #Number of filters and convolutional layers
    nf=0;layers=0
    #Dilation rate sum
    dilation=0
    #Compression rate(\theta)
    tetha=0
    n1=0;n2=0;
    for m in model.downs[0].modules():
        if isinstance(m, nn.Conv2d):
            kx+=m.kernel_size[0]
            ky+=m.kernel_size[1]
            nf+=m.out_channels
            layers+=1
            dilation+=m.dilation[0]
        if isinstance(m, DenseBlockConnection):
            tetha+=m.tetha
        if isinstance(m, MyModule_AddSubCat):
            n1+=m.n1
            n2+=m.n2        
            
    return kx, ky, nf, layers, dilation, tetha, n1, n2
    

def get_features(individual):
    tree_feat=np.array([individual.height, len(individual)])
    
    model = make_model(individual, in_channels, pset)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = np.array([params])
    
    # conv_layers, filters = count_LayersAndFilters(model)
    kx, ky, nf, layers, dilation, tetha, n1, n2 = sum_terminals(model)
    
    count_f=count_functions(individual)
    functions=np.array(list(count_f.values()))
    
    parents_feats = np.array([individual.dice_p1, individual.dice_p2, individual.dice_m])
    if (parents_feats==0.0).all():
        parents_feats = np.array([np.nan,np.nan,np.nan])
        
    return np.concatenate((tree_feat,params,functions,np.array([kx, ky, nf, layers, dilation, tetha, n1, n2]),parents_feats))

def get_features2(ind):
    # Size and depth of the trees
    tree_feats=np.array([ind.height, len(ind)])
    
    # extract the values of the constants from the individual
    consts = filter(lambda x: isinstance(x, gp.Ephemeral), ind)
    const_values = list(map(lambda x: float(x.name), consts))
    # compute statistics on constants
    const_avg = sum([abs(x) for x in const_values])/len(const_values) if const_values else 0
    const_max = max(const_values) if const_values else -1e6
    const_min = min(const_values) if const_values else +1e6
    const_distinct = len(set(const_values))/len(const_values) if const_values else 1
    const_feats = np.array([len(const_values)/len(ind), const_avg, const_max, const_min, const_distinct])
    
    # get the names used in the individual, use names of constants instead of their values
    ind_names = list(map(lambda x: x.__class__.__name__ if isinstance(x, gp.Ephemeral) else x.name, ind))
    # count how many times each name is in the individual
    counts = collections.Counter(ind_names)
    # Primitive Names
    primitive_names=list(pset.context.keys())[1:]
    terminal_names = list(map(lambda x: x.name if isinstance(x, gp.Terminal) else x.__name__,
                              itertools.chain.from_iterable(pset.terminals.values())))
    # frequency of each primitiveprimitives
    freqprimitives_feats = np.array([counts[x]/len(ind) for x in primitive_names])
    # frequency of each terminal
    freqterminals_feats = np.array([counts[x]/len(ind) for x in terminal_names])
    
    #Trainable parameters
    model = make_model(ind, in_channels, pset)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #Sum of the kernel size, number of filter, conv layers, dilation rate, compression rate
    #and (n1,n2) for the add and sub functions
    kx, ky, nf, layers, dilation, tetha, n1, n2 = sum_terminals(model)
    model_feats = np.array([params, kx, ky, nf, layers, dilation, tetha, n1, n2])
    
    # parents_feats = np.array([ind.dice_p1, ind.dice_p2, ind.dice_m])
    # if (parents_feats==0.0).all():
    #     parents_feats = np.array([np.nan,np.nan,np.nan])
    
    return np.concatenate((tree_feats, const_feats, freqprimitives_feats, freqterminals_feats, model_feats))
    # return np.concatenate((tree_feats, const_feats, freqprimitives_feats, freqterminals_feats, model_feats, parents_feats))

def surrogate_train(train, train_fit):
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    imputer = SimpleImputer(strategy='median')
    # imputer = Imputer(strategy='median')
    
    reg = Pipeline([('imputer', imputer),
                    ('regressor', rf)])
    
    reg.fit(train, train_fit)
    return reg

def evaluate_surrogate(individual, reg):
    # encoding=get_features(individual)
    encoding=get_features2(individual)
    dice = reg.predict([encoding])[0]
    model = toolbox.make_model(individual, in_channels)
    complexity, params = evaluate_NoParameters(model, dloaders.IN_CHANNELS, max_params, pset)
    fit = (1 - w)*dice + w*complexity
    return fit, dice, params

#Train surrogate model. Evaluate a individuakl in the surrogate model
# toolbox.register("get_features", get_features)
toolbox.register("get_features", get_features2)
toolbox.register("train_surrogate", surrogate_train)
toolbox.register("evaluate_surrogate", evaluate_surrogate)

#%%%Initial Sampling
def sampling(n, nepochs, lossfn, lr, 
             loaders, pset, 
             device, ruta, verbose_train):
    #Generate population
    train_pop=toolbox.population(n)
    #Evaluate dice
    dices = []
    
    for ind in train_pop:
        model=toolbox.make_model(ind, loaders.IN_CHANNELS)
        metrics, _ = evaluate_Segmentation(model, nepochs, lossfn, lr,
                                           loaders, device, ruta, verbose_train)
        dices.append(np.mean(metrics["dices"]))
    #Get features
    train_set = [get_features(ind) for ind in train_pop]
    
    return train_pop, train_set, dices

def cache_archive(n):
    #Generate population
    train_pop=toolbox.population(n)
    
    cache={}
    archive=np.empty([])
    
    for ind in train_pop:
        key = toolbox.identifier(ind)
        ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
        
        #Add to cache when original objective function is used
        if key not in cache:
            #Add to cache
            cache[key]=ind
            
            #Get features of individual
            features=toolbox.get_features(ind)
            labels=np.array([ind.dice])
            features=np.expand_dims(np.array(features),axis=0)
            labels=np.expand_dims(np.array(labels),axis=0)
            new_instance=np.concatenate((features,labels), axis=1)
            
            #Add it to archive
            if archive.shape==():
                archive = new_instance
            else:
                archive = np.concatenate((archive, new_instance), axis=0)
    
    return cache, archive
    

#%%Storage results
"""Create folder to storage"""
foldername="SEA-GS-170723"
# path='/scratch/202201016n'
path="C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/surrogate_test"
ruta=path+"/surrogate/"+str(foldername)
if not os.path.exists(ruta):
    os.makedirs(ruta)
    
np.random.seed(10)
random.seed(10)
    
#%%Data
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

#%%Evolutionary parameters
pop_size = 10
ngen = 2
cxpb = 0.8
mutpb = 0.19
nelit = 1
tz = 7
gen_update = 4
p = pop_size//10
m = 1000
mstats=statics_()
hof = tools.HallOfFame(nelit)
checkpoint_name = False#'checkpoint_evo.pkl'#False
verbose_evo=False
max_params = 31038000
w = 0.01
evolutionary_parameters = {'population_size':pop_size, 'n_gen': ngen, 
                            'cxpb': cxpb, 'mutpb':mutpb, 'n_elit':nelit, 'tournament_size':tz,
                            'max_params':max_params, 'w': w
                            }

#%%Training parameters
nepochs = 1
alpha = 0.5
beta = 0.4
lossfn = ComboLoss(alpha=alpha, beta=beta)
lr = 0.0001

training_parameters = {'num_epochs':nepochs, 'loss_f':lossfn, 'learning_rate':lr}

verbose_train=False

toolbox.register("evaluate", evaluation, nepochs=nepochs, lossfn=lossfn, lr=lr, max_params=max_params, w=w, 
                             loaders=dloaders, pset=pset, device="cuda:0", ruta=ruta, verbose_train=verbose_train)
toolbox.register("sampling", sampling, nepochs=nepochs, lossfn=lossfn, lr=lr,
                             loaders=dloaders, pset=pset, device="cuda:0", ruta=ruta, 
                             verbose_train=verbose_train)
toolbox.register("cache_archive", cache_archive, n=200)
#%%Run Algorithm
pop, log, archive, cache = eaNASGP_Net2(pop_size, toolbox, cxpb, mutpb, ngen, nelit, gen_update, p, m,
                                        ruta, checkpoint_name,
                                        stats=mstats, halloffame=hof, verbose_evo=verbose_evo)

#%%Check Diversity 1
individuals_cache = list(cache.values())
for i in range(len(individuals_cache)):
    ind_ref = individuals_cache[i]
    for j in tqdm(range(len(individuals_cache))):
        if j==i:
            pass
        else:
            ind = individuals_cache[j]
            if ind==ind_ref:
                print("Same Tree", ind, ind_ref)
                feats_ref=archive[i,:-1] #toolbox.get_features(ind_ref)
                feats = archive[j,:-1]#toolbox.get_features(ind)
                if (feats_ref==feats).all():
                    print("Same Features", feats, feats_ref)
                    
                    key_ref = toolbox.identifier(ind_ref)
                    key = toolbox.identifier(ind)
                    print("Fit in cache\t", cache[key_ref].dice, cache[key].dice)
                    
                    fit_ref = archive[i,-1]#cache[key_ref].dice
                    fit = archive[j,-1]#cache[key].dice
                    print("Fit in archive\t", fit_ref, fit)
                    
                    if fit_ref == fit:
                        print("Are Equal!", fit, fit_ref)
                    else:
                        print("Are Different", fit, fit_ref, "\n\n")
            else:
                feats_ref=archive[i,:-1] #toolbox.get_features(ind_ref)
                feats = archive[j,:-1]#toolbox.get_features(ind)
                if (feats_ref==feats).all():
                    print("Different Individuals\n", ind, '\n', ind_ref)
                    print("Same Features", feats, feats_ref)
                
                    key_ref = toolbox.identifier(ind_ref)
                    key = toolbox.identifier(ind)
                    print("Fit in cache\t", cache[key_ref].dice, cache[key].dice)
                    
                    fit_ref = archive[i,-1]#cache[key_ref].dice
                    fit = archive[j,-1]#cache[key].dice
                    print("Fit in archive\t", fit_ref, fit)
                    
                    if fit_ref == fit:
                        print("Are Equal!", fit_ref, fit)
                    else:
                        print("Are Different", fit_ref, fit, "\n\n")

#%%%Save statistics
"""Show and save Statics as .png and .csv"""
save_statics(log, ruta)
show_statics(log, ruta)

#%%%Best individual
"""Plot Best individual"""
best=tools.selBest(pop,1)[0]

#%%%Evo Details
"""Save details about evolutionary process"""
saveEvolutionaryDetails(evolutionary_parameters, best,
                        log.select("nevals"), log.select("time"),
                        filename=ruta+'/evolution_details.txt')

#%%%Function frequency
functionAnalysis(pop,10,pset,ruta)

#%%Train again the best architecture for reliable compatarion with U-Net
best_model=toolbox.clone(best)
model=toolbox.make_model(best_model, in_channels=in_channels)

"""Train, val and test loaders"""
train_loader, _= dloaders.get_train_loader()
val_loader, _  = dloaders.get_val_loader()
test_loader, _ = dloaders.get_test_loader()

"""Optimizer"""
optimizer = optim.Adam(model.parameters(), lr=lr)

"""Epocas de re-entrenamiento"""
num_epochs=100

"""Device"""
device='cuda:0'

"""For save images or model"""
save_model_flag=True
save_images_flag=True
load_model=False

#%%%Train, validate and test
"""Train and valid model"""
train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
    model, train_loader, val_loader,
    num_epochs, optimizer, lossfn,
    device, load_model, save_model_flag,
    ruta=ruta, verbose=True
    )

#%%%Test
"""Test model"""
dices, ious, hds = test(test_loader, model, lossfn,
                        save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)

# best_model=toolbox.clone(best)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
best_model.fitness.values = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params),
best_model.dice = np.mean(dices)
best_model.params = params

best_model.train_loss = train_loss
best_model.valid_loss = valid_loss
best_model.train_dice = train_dice
best_model.valid_dice = valid_dice

best_model.dices = dices
best_model.ious = ious
best_model.hds = hds

#%%%NNo. parameters
"""No of parameters"""
model = model.to(device)
model_stats=summary(model, (1, in_channels, 256, 256), verbose=0)
summary_model = str(model_stats)

#%%%Retrain Details
saveTrainingDetails(training_parameters, dloaders,
                    best_model, summary_model,
                    filename=ruta+'/Retrain_best_details.txt')

#%%Save execution
save_execution(ruta, foldername+'.pkl', pop, log, archive, cache, best_model)