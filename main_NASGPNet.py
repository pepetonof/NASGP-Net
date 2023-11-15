# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 04:02:23 2022

@author: josef
"""

#%% Import libraries
import os
import torch
# import pandas as pd
import numpy as np
from torchinfo import summary
# import torch.multiprocessing as mp
import torch.optim as optim
from deap import tools
from deap import creator, base
import deap.gp as gp
# import time
import operator

import gp_restrict
import gp_tree
from strongGPDataType import (moduleTorch, moduleTorchL, moduleTorchSe, moduleTorchCn, moduleTorchCt, moduleTorchP,
                              outChConv, outChSConv, kernelSizeConv, dilationRate,
                              tetha,wArithm)
from operators.functionSet import (convolution, sep_convolution,
                             res_connection, dense_connection,
                             se,
                             add, sub, cat,
                             maxpool, avgpool)
from toolbox_functions import (make_model, evaluation,
                               save_ind, save_graphtv, save_graphtvd,
                               identifier)

from model.loss_f import ComboLoss
from model.train_valid import train_and_validate
from model.predict import test
from data.loader import loaders
# from algorithm import NASGP_Net
from algorithm_NASGPNet import eaNASGPNet
from utils.save_utils import saveEvolutionaryDetails, saveTrainingDetails, save_execution
from utils.deap_utils import statics_, log2csv, functionAnalysis, show_statics

import data.dataSplit as dataSplit
# import data.dataStatic as dataStatic

# from objective_functions import evaluate_Segmentation, evaluate_NoParameters

# foldername='test11'

#%%Pset
pset = gp_tree.PrimitiveSetTyped("main", [moduleTorch], moduleTorchP)

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
                dice=float, iou=float, hd=float, hd95=float, 
                #nds=float,
                params=int)

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)
# toolbox.register('evaluationMP', evaluationMP, loaders=loaders, pset=pset)
# toolbox.register('evaluate', evaluationMP, loaders=loaders, pset=pset)

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

# foldername='testMultiClassHIPPOCAMPUS2'

#%%GPMAIN
def GPMain(foldername):
    """Create folder to storage"""
    # path='/scratch/202201016n'
    path= "C:/Users/josef/serverBUAP/corridas"
    ruta=path+"/reply/"+str(foldername)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    
    #%% Data
    # path_images='/home/202201016n/serverBUAP/datasets/images_DHIPPO'
    path_images = 'C:/Users/josef/serverBUAP/datasets/images_DHIPPO'
    in_channels = 1
    out_channels= 2
    
    """Get train, valid, test set and loaders"""
    train_set, valid_set, test_set = dataSplit.get_data(0.7, 0.15, 0.15, path_images)
    # train_set, valid_set, test_set = dataStatic.get_data(path_images)
    
    IMAGE_HEIGHT = 64#288 
    IMAGE_WIDTH = 64#480
    NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders
    BATCH_SIZE = 16
    
    dloaders = loaders(train_set, valid_set, test_set, batch_size=BATCH_SIZE, 
                     image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, 
                     in_channels=in_channels, out_channels=out_channels)
       
    #%% Evolutionary process, evo-statics and parameters
    #Evolutionary parameters
    pz = 10
    ng = 5
    cxpb = 0.8
    mutpb = 0.19
    nelit = 1 
    tz = 7
    mstats=statics_()
    hof = tools.HallOfFame(nelit)
    checkpoint_name = False #'checkpoint_evo.pkl'#False
    verbose_evo=False
    max_params = 31038000
    w = 0.01
    evolutionary_parameters = {'population_size':pz, 'n_gen': ng, 
                               'cxpb': cxpb, 'mutpb':mutpb, 'n_elit':nelit, 'tournament_size':tz,
                               'max_params':max_params, 'w': w
                               }
    
    
    #Training_parameters
    nepochs = 2
    alpha = 0.5
    beta = 0.4
    # lossfn = ComboLoss(alpha=alpha, beta=beta, average='micro', eps=1.0)
    # lossfn = DiceLoss(average='macro', eps=1)
    # lossfn = ComboLoss(alpha=alpha, beta=beta, average='micro')
    lossfn = ComboLoss(alpha=alpha, beta=beta, average="micro", include_background=True)
    lr = 0.0001
    training_parameters = {'num_epochs':nepochs, 'loss_f':lossfn, 'learning_rate':lr}
    verbose_train=True
    
    toolbox.register("evaluate", evaluation, nepochs=nepochs, lossfn=lossfn, lr=lr, max_params=max_params, w=w, 
                                 loaders=dloaders, pset=pset, device="cuda:0", ruta=ruta, verbose_train=verbose_train)
    
    #randomSeeds=int(foldername[-1])
    #random.seed(randomSeeds)
    
    #%%Run algorithm
    pop, log, cache = eaNASGPNet(pop_size = pz, toolbox = toolbox, 
                                  cxpb = cxpb, mutpb = mutpb, 
                                  ngen = ng, nelit = nelit, 
                                  ruta = ruta, checkpoint_name = checkpoint_name,
                                  stats=mstats, halloffame=hof, verbose_evo=verbose_evo)
    
    #%%Save statistics
    """Show and save Statics as .png and .csv"""
    # save_statics(log, ruta)
    log2csv(log, mstats, ruta)
    show_statics(log, ruta)
    
    #%%Best individual
    """Plot Best individual"""
    best=tools.selBest(pop,1)[0]
    #Testing multiclass
    # best=toolbox.individual()
    
    #%%Evo Details
    """Save details about evolutionary process"""
    saveEvolutionaryDetails(evolutionary_parameters, best,
                            log.select("nevals"), log.select("time"),
                            filename=ruta+'/evolution_details.txt')
    
    #%%Function frequency
    functionAnalysis(pop,10,pset,ruta)
    
    #%%Train again the best architecture for reliable compatarion with U-Net
    #Parameters:
    IMAGE_HEIGHT = 64#288 
    IMAGE_WIDTH = 64#480
    NUM_WORKERS = 0 if torch.cuda.is_available() else 0 #Also used for dataloaders
    BATCH_SIZE = 2
    
    dloaders = loaders(train_set, valid_set, test_set, batch_size=BATCH_SIZE, 
                     image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, 
                     in_channels=in_channels, out_channels = out_channels)
    
    best_model=toolbox.clone(best)
    model=toolbox.make_model(best_model, in_channels, out_channels)
    
    """Train, val and test loaders"""
    train_loader, _= dloaders.get_train_loader()
    val_loader, _  = dloaders.get_val_loader()
    test_loader, _ = dloaders.get_test_loader()
    
    """Optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    """Epocas de re-entrenamiento"""
    nepochs=10
    
    """Device"""
    device='cuda:0'
    
    """For save images or model"""
    save_model_flag=True
    save_images_flag=True
    load_model=False
    
    """Print the process during training"""
    verbose_train=True
    
    #%%Train, validate and test
    """Train and valid model"""
    train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
        model, train_loader, val_loader,
        nepochs, optimizer, lossfn,
        device, load_model, save_model_flag,
        ruta=ruta, verbose=verbose_train,
        )
    
    """Test model"""
    dices, ious, hds, hds95 = test(test_loader, model,
                            save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #%%Assign attributes
    best_model.fitness.values = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params),
    
    #Average metrics
    best_model.dice = np.mean(dices)
    best_model.iou = np.mean(ious)
    best_model.hd = np.mean(hds)
    best_model.hd95 = np.mean(hds95)
    # best_model.nsd = np.mean(nsds)
    best_model.params = np.mean(params)
    
    #Metrics on each element in the batch
    best_model.dices = dices
    best_model.ious = ious
    best_model.hds = hds
    best_model.hd95 = hds95
    # best_model.nsd = nsds
    best_model.params = params
    
    #loss and dice during training and validation
    best_model.train_loss = train_loss
    best_model.valid_loss = valid_loss
    best_model.train_dice = train_dice
    best_model.valid_dice = valid_dice
    
    #%%Dice and Loss Graphs of training and validation
    # """Coloca y guarda la gráfica de entrenamiento del mejor individuo"""
    # toolbox.save_graphtv(best_model, ruta=ruta, filename='ReTrainValidationLoss')
    # toolbox.save_graphtvd(best_model, ruta=ruta, filename='ReTrainValidationDice')
    
    #%%No. parameters and Summary
    """No of parameters"""
    model = model.to(device)
    model_stats=summary(model, (1, in_channels, IMAGE_HEIGHT, IMAGE_WIDTH), verbose=0)
    summary_model = str(model_stats)
    
    #%%Retrain Details
    saveTrainingDetails(training_parameters, dloaders, 
                        best_model, summary_model,
                        filename=ruta+'/Retrain_best_details.txt')
    
    #%%Save execution: pop., log, cache, best model as .pkl file
    save_execution(ruta, foldername+'.pkl', pop, log, cache, best_model)
    
    del model
    
    return log, pop, best_model
    
if __name__=='__main__':
    # mp.set_start_method('forkserver')
    log, pop, best = GPMain('MSD-Hippo-MicroTrue')
