# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:35:40 2023

@author: josef
"""

import sys
sys.path.append( 'C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/NASGP-Net' )


#%% Import libraries
import os
import torch
import pandas as pd
import numpy as np
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
from toolbox_functions import (make_model, evaluationMP, 
                               save_ind, save_graphtv, save_graphtvd,
                               identifier)

from model.loss_f import ComboLoss
from model.train_valid import train_and_validate
from model.predict import test
from data.loader import loaders
# from algorithm import NASGP_Net
from surrogate_algorithm import NASGP_Net
from utils.save_utils import saveEvolutionaryDetails, saveTrainingDetails, save_execution
from utils.deap_utils import statics_, save_statics, show_statics, functionAnalysis

import data.dataSplit as dataSplit
import data.dataStatic as dataStatic

from sklearn.ensemble import RandomForestRegressor

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
                dices=list, ious=list, hds=list,
                train_loss=list, valid_loss=list,
                params=int, dice=float,
                )

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)
toolbox.register('evaluate', evaluationMP)

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
    
toolbox.register("save_graphtvd", save_graphtvd)    
toolbox.register("save_graphtv", save_graphtv)

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

def get_features(individual):
    tree_feat=np.array([individual.height, len(individual)])
    parent_fit=np.random.uniform(low=0.5, high=1, size=(2,))
    parent_fit=np.append(parent_fit, np.mean(parent_fit))
    
    model = make_model(individual, in_channels, pset)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = np.array([params])
    
    count_f=count_functions(individual)
    functions=np.array(list(count_f.values()))
    
    return np.concatenate((tree_feat,params,functions))

def surrogate_train(train, train_fit):
    reg = RandomForestRegressor(n_estimators=100, random_state=0)
    reg.fit(train, train_fit)
    return reg

def evaluate_surrogate(individual, reg):
    encoding=get_features(individual)
    fitness_vals = reg.predict([encoding])
    return fitness_vals[0]

toolbox.register("evaluate_surrogate", evaluate_surrogate)
toolbox.register("train_surrogate", surrogate_train)

#%%%Initial sampling and evaluation in original model
def evaluate_training_set(train):
    train_fit=[]
    for idx, el in enumerate(train):
        # fit,_,_,=toolbox.evaluate(train[idx])
        fit,_,_,=toolbox.evaluate(train[idx], self.nepochs, self.loss_fn, self.lr, 
                                        self.max_params, self.w, 
                                        self.loaders, self.pset, 'cuda:'+str(0), self.ruta, self.verbose_train)
        train_fit.append(fit)
    return train_fit

def sampling(n):
    train=toolbox.population(n)
    # train = [get_features(ind) for ind in train]
    fitness=evaluate_training_set(train)
    train_feats= [get_features(ind) for ind in train]
    return train, train_feats, fitness

toolbox.register("sampling", sampling)

#%%Storage results
"""Create folder to storage"""
foldername="SEA-Test"
# path='/scratch/202201016n'
path="C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/surrogate_test"
ruta=path+"/surrogate/"+str(foldername)
if not os.path.exists(ruta):
    os.makedirs(ruta)

#%%Data
path_images = 'C:/Users/josef/serverBUAP/datasets/images_DRIVE'
in_channels=3

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

#%% Evolutionary process, evo-statics and parameters
#Evolutionary parameters
# pz = 10
# ng = 5
# cxpb = 0.8
# mutpb = 0.19
# nelit = 1 
# tz = 3
# max_params = 31038000
# w = 0.01
# evolutionary_parameters = {'population_size':pz, 'n_gen': ng, 
#                            'cxpb': cxpb, 'mutpb':mutpb, 'n_elit':nelit, 'tournament_size':tz,
#                            'max_params':max_params, 'w': w
#                            }


# #Training_parameters
# ##alpha=0.5, beta=0.4 for MRI and CT
# num_epochs = 5
# alpha=0.5
# beta=0.4
# loss_fn = ComboLoss(alpha=alpha, beta=beta)
# lr = 0.0001
# training_parameters = {'num_epochs':num_epochs, 'loss_f':loss_fn, 'learning_rate':lr}
# mstats=statics_()
# hof = tools.HallOfFame(10)
# checkpoint = False#'checkpoint_evo.pkl'#False

# #randomSeeds=int(foldername[-1])
# #random.seed(randomSeeds)

# #%%Run algorithm
# ea = NASGP_Net(evolutionary_parameters, training_parameters,
#                 toolbox = toolbox, pset = pset, loaders = dloaders,
#                 stats = mstats, halloffame = hof, verbose_evo=False, verbose_train=False,
#                 checkpoint = checkpoint, ruta = ruta, foldername=foldername
#                 )

# pop, log = ea.run()

#%%Save detial about algorithm
# #%%%Save statistics
# """Show and save Statics as .png and .csv"""
# save_statics(log, ruta)
# show_statics(log, ruta)

# #%%%Best individual
# """Plot Best individual"""
# best=tools.selBest(pop,1)[0]

# #%%%Evo Details
# """Save details about evolutionary process"""
# saveEvolutionaryDetails(evolutionary_parameters, best,
#                         ea.no_evs, ea.delta_t,
#                         filename=ruta+'/evolution_details.txt')

# #%%%Function frequency
# functionAnalysis(pop,10,pset,ruta)

# #%%$Train again the best architecture for reliable compatarion with U-Net
# best_model=toolbox.clone(best)
# model=toolbox.make_model(best_model, in_channels=in_channels)

# """Train, val and test loaders"""
# train_loader, _= dloaders.get_train_loader()
# val_loader, _  = dloaders.get_val_loader()
# test_loader, _ = dloaders.get_test_loader()

# """Optimizer"""
# optimizer = optim.Adam(model.parameters(), lr=lr)

# """Epocas de re-entrenamiento"""
# num_epochs=10

# """Device"""
# device='cuda:0'

# """For save images or model"""
# save_model_flag=True
# save_images_flag=True
# load_model=False

# #%%%Train, validate and test
# """Train and valid model"""
# train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
#     model, train_loader, val_loader,
#     num_epochs, optimizer, loss_fn,
#     device, load_model, save_model_flag,
#     ruta=ruta, verbose=True
#     )

# """Test model"""
# dices, ious, hds = test(test_loader, model, loss_fn,
#                         save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)

# #%%%Attributes assigned
# # best_model=toolbox.clone(best)
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# best_model.fitness.values = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params),
# best_model.dice = np.mean(dices)
# best_model.params = params

# best_model.train_loss = train_loss
# best_model.valid_loss = valid_loss
# best_model.train_dice = train_dice
# best_model.valid_dice = valid_dice

# best_model.dices = dices
# best_model.ious = ious
# best_model.hds = hds

# #%%Dice and Loss Graphs of training
# """Coloca y guarda la gráfica de entrenamiento del mejor individuo"""
# toolbox.save_graphtv(best_model, ruta=ruta, filename='ReTrainValidationLoss')
# toolbox.save_graphtvd(best_model, ruta=ruta, filename='ReTrainValidationDice')

# #%%NNo. parameters
# """No of parameters"""
# model = model.to(device)
# model_stats=summary(model, (1, in_channels, 256, 256), verbose=0)
# summary_model = str(model_stats)

# #%%Retrain Details
# saveTrainingDetails(training_parameters, dloaders, 
#                     best_model, summary_model,
#                     filename=ruta+'/Retrain_best_details.txt')

# save_execution(ruta, foldername+'.pkl', pop, log, best_model)


#%%Main for 1st Test
#%%%Create training set
train_size=20
train_pop, train_feats, fitnesses = sampling(train_size)

#%%%Train surrogate model
rf_reg=surrogate_train(train_feats, fitnesses)

#%%%Create randomly an individual
np.random.seed(48)
individual=toolbox.individual()
print(individual)
count_f=count_functions(individual)
print(count_f)
fets=get_features(individual)
print(fets)


#%%%Evaluate in the original objective function
start_time = time.time()
fit_real=toolbox.evaluate(individual)
print("--- %s seconds ---" % (time.time() - start_time))

#%%%Evaluate in the subrogate model
start_time = time.time()
fit_surr=toolbox.evaluate_surrogate(individual, reg=rf_reg)
print("--- %s seconds ---" % (time.time() - start_time))

#%%%Calculate the metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
test_pop, test_feats, t_fitnesses = sampling(5)
preds=np.array([evaluate_surrogate(ind, rf_reg) for ind in test_pop])
#%%%%
t_fitnesses=np.array([f[0] for f in t_fitnesses])
r2=r2_score(t_fitnesses, preds)
mae=mean_absolute_error(t_fitnesses, preds)
mse=mean_squared_error(t_fitnesses, preds)
ratio=preds/t_fitnesses ##-

print("R2:\t",r2)
print("MAE:\t",mae)
print("MSE:\t",mse)
print("Ratio:\t",ratio)