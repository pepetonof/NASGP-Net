# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 04:02:23 2022

@author: josef
"""

#%% Import libraries
import os
from deap import tools
from deap import creator, base
import deap.gp as gp
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
from toolbox_functions import (make_model, evaluation, evaluation_cv,
                               save_ind, save_graphtv, save_graphtvd,
                               identifier)

# from model.loss_f import ComboLoss
from losses.loss_functions import DiceLoss

from algorithm_NASGPNet import eaNASGPNet
from utils.save_utils import saveEvolutionaryDetails, save_execution
from utils.deap_utils import statics_, log2csv, functionAnalysis, show_statics

import data.dataSplit as dataSplit
import data.dataStatic as dataStatic

from data.dataset import Dataset2D#, Dataset3D22D, Dataset3D
from metrics.segmentation_metrics import DiceMetric, IoUMetric, HDMetric, NSDMetric

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

# Se,Se; Se,L; L,Se;
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

#Feature Connection Layer Optional
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
                params=int)

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)

# toolbox.register("select", tools.selTournament, tournsize=3)
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

# foldername='test'

#%%GPMAIN
def GPMain(foldername):
    """Create folder to storage"""
    # path='/scratch/202201016n'
    path= "C:/Users/josef/serverBUAP/corridas"
    ruta=path+"/cross_validation/"+str(foldername)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    
    #%% Data
    # path_images='/home/202201016n/serverBUAP/datasets/images_DHIPPO'
    # path_images = 'C:/Users/josef/serverBUAP/datasets/images_ISIC'
    path_images = "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/comparison-datasets/folds"
    in_channels = 1
    out_channels = 2
    dataset_type = Dataset2D
    no_classes_msk = 2
    image_height = 512
    image_width = 512
    batch_size = 2

    """Split Data (Percent or Static, 70-15-15)"""
    # train_set, valid_set, test_set = dataSplit.get_data(0.7, 0.15, 0.15, path_images,_format='.png')
    # train_set, valid_set, test_set = dataStatic.get_data(path_images, val_size=0.1, _format='.png')
    train_set, valid_set, test_set = dataStatic.get_data_folds(path_images, val_size=0.1, _format='.png')
    #%% Evolutionary process, evo-statics and parameters
    #Evolutionary parameters
    pz = 5
    ng = 2
    cxpb = 0.5
    mutpb = 0.49
    nelit = 1 
    tz=7
    mstats=statics_()
    hof = tools.HallOfFame(nelit)
    checkpoint_name = False#'checkpoint_evo.pkl'#,False
    verbose_evo = False
    max_params = 31038000
    w = 0.3
    k_folds = False

    #Training_parameters
    nepochs = 1
    #alpha = 0.5
    #beta = 0.7
    #lossfn = ComboLoss(alpha=alpha, beta=beta, average="micro", include_background=True)
    lossfn = DiceLoss(average='macro', include_background=False, softmax=False, eps=1e-8)

    spacing_mm = (1,1,0)
    metrics = [
        DiceMetric(average='macro', include_background=False, softmax=False, eps=1e-8),
        IoUMetric(average='macro', include_background=False, softmax=False, eps=1e-8),
        HDMetric(average='macro', include_background=False, softmax=False, spacing_mm=spacing_mm),
        NSDMetric(average='macro', include_background=False, softmax=False, spacing_mm=spacing_mm, tolerance=1),
        ]
    lr = 0.0001
    tolerance = 3
    verbose_train=True
    device='cuda:0'
    save_model=False
    save_images=False
    save_datafolds=False
    save_data=False
    
    toolbox.register("select", tools.selTournament, tournsize=tz)
    toolbox.register("evaluate", evaluation_cv, 
                                  nepochs=nepochs, 
                                  tolerance=tolerance, 
                                  lossfn=lossfn,
                                  metrics=metrics,
                                  lr=lr, 
                                  dataset_type = dataset_type,
                                  no_classes_msk = no_classes_msk,
                                  in_channels=in_channels, 
                                  out_channels=out_channels,
                                  batch_size = batch_size,
                                  image_height = image_height,
                                  image_width = image_width,
                                  max_params=max_params, w=w, 
                                  k_folds=k_folds,
                                  train_set=train_set, 
                                  valid_set=valid_set, 
                                  test_set=test_set,
                                  pset = pset, 
                                  device=device, 
                                  ruta=ruta, 
                                  verbose_train=verbose_train,
                                  save_model=save_model,
                                  save_images=save_images,
                                  save_datafolds=save_datafolds)

    # toolbox.register("evaluate", evaluation, 
    #                              nepochs=nepochs, 
    #                              tolerance=tolerance, 
    #                              lossfn=lossfn,
    #                              metrics=metrics,
    #                              lr=lr, 
    #                              dataset_type = dataset_type,
    #                              no_classes_msk = no_classes_msk,
    #                              in_channels=in_channels, 
    #                              out_channels=out_channels,
    #                              batch_size = batch_size,
    #                              image_height = image_height,
    #                              image_width = image_width,
    #                              max_params=max_params, w=w, 
    #                              train_set=train_set, 
    #                              valid_set=valid_set, 
    #                              test_set=test_set,
    #                              pset = pset, 
    #                              device=device, 
    #                              ruta=ruta, 
    #                              verbose_train=verbose_train,
    #                              save_model=save_model,
    #                              save_images=save_images,
    #                              save_data=save_data)
    
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
    
    #%%Evo Details
    """Save details about evolutionary process"""
    evolutionary_parameters = {'population_size':pz, 'n_gen': ng, 
                               'cxpb': cxpb, 'mutpb':mutpb, 'n_elit':nelit, 'tournament_size':tz,
                               'max_params':max_params, 'w': w
                               }
    saveEvolutionaryDetails(evolutionary_parameters, best,
                            log.select("nevals"), log.select("time"),
                            filename=ruta+'/evolution_details.txt')
    
    #%%Function frequency
    functionAnalysis(pop,10,pset,ruta)
    
    #%%%change parameters values
    nepochs=5
    tolerance=3
    verbose_train=True
    save_model=True
    save_images=True
    save_data=True

    #%%%Re evaluate best individual and save data
    fit, dice, iou, hds, hds95, nsds, params = toolbox.evaluate(best,
                                                                nepochs=nepochs,
                                                                tolerance=tolerance,
                                                                verbose_train=verbose_train,
                                                                save_model=save_model,
                                                                save_images=save_images,
                                                                save_data=save_data)

    #%%%Save execution
    save_execution(ruta, foldername+'.pkl', pop, log, cache, best)
    
    return log, pop, best
    
if __name__=='__main__':
    # mp.set_start_melthod('forkserver')
    log, pop, best = GPMain('MAMMO-PRUEBA')