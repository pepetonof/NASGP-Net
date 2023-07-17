# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:12:19 2023

@author: josef
"""

#%% Import libraries
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

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)
# toolbox.register('evaluationMP', evaluationMP, loaders=loaders, pset=pset)
# toolbox.register('evaluate', evaluationMP, loaders=loaders, pset=pset)

#%%Sel Torunament based on non-dominance
def selTournament(individuals, k, tournsize=2):
  chosen=[]
  for i in range(k):
    aspirants = random.sample(individuals, tournsize)
    r=dominate_relation(aspirants[0], aspirants[1])
    if r==0:
      chosen.append(aspirants[0])
    elif r==1:
      chosen.append(aspirants[1])
    elif r==2:
      chosen.append(random.choice(aspirants))
  return chosen

# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", selTournament)
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

#%%GPMAIN
def GPMain(foldername, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    """Create folder to storage"""
    # foldername="SEA-GS-060623-COVID_SM-P0-2"
    # path='/scratch/202201016n'
    path= "C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/EEs/MO-1/Proyecto"
    ruta=path+"/first_attempt/"+str(foldername)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    
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
       
    #%% Evolutionary process, evo-statics and parameters
    #Evolutionary parameters
    pz = 10
    ng = 10
    cxpb = 0.8
    mutpb = 0.19
    nelit = 1 
    tz = 7
    mstats=statics_()
    hof = tools.HallOfFame(nelit)
    checkpoint_name = False#'checkpoint_evo.pkl'#False
    verbose_evo=False
    max_params = 31038000
    w = 0.01
    
    evolutionary_parameters = {'population_size':pz, 'n_gen': ng, 
                               'cxpb': cxpb, 'mutpb':mutpb, 'n_elit':nelit, 'tournament_size':tz,
                               'max_params':max_params, 'w': w
                               }
    
    
    #Training_parameters
    nepochs = 1
    alpha = 0.5
    beta = 0.4
    lossfn = ComboLoss(alpha=alpha, beta=beta)
    lr = 0.0001
    training_parameters = {'num_epochs':nepochs, 'loss_f':lossfn, 'learning_rate':lr}
    verbose_train=False
    
    toolbox.register("evaluate", evaluationMO, nepochs=nepochs, lossfn=lossfn, lr=lr, max_params=max_params,
                                 loaders=dloaders, pset=pset, device="cuda:0", ruta=ruta, verbose_train=verbose_train)
    
    #uGA parameters
    sz_e = 50
    sz_m = 200
    mnr_p  = 0.3
    sz_up = 10
    its_uga = 2
    its = 100
    div_agrid = 25
    rep_cycle = 10
    checkpoint = False#"checkpoint.pkl" #False
    uga_parameters = {"size_external":sz_e, "size_mempob":sz_m, "memnr_p": mnr_p, "size_upop":sz_up,
                     "its_uga":its_uga, "its":its, "div_agrid":div_agrid, "rep_cycle":rep_cycle}
    
    #randomSeeds=int(foldername[-1])
    #random.seed(randomSeeds)
    
    #%%Run algorithm
    # pop, log, cache = eaNASGPNet(pop_size = pz, toolbox = toolbox, cxpb = cxpb, mutpb = mutpb, ngen = ng, nelit = nelit, 
    #                        # gen_update, p, m,
    #                        ruta = ruta, checkpoint_name = checkpoint_name,
    #                        stats=mstats, halloffame=hof, verbose_evo=verbose_evo)
    
    pop = toolbox.population(n=sz_m)
    M, E, g_log = eaMONASGPNet_CHKP(pop=pop, upop_size=sz_up, percmem_nr=mnr_p,
                               extmem_size=sz_e, its=its, its_u=its_uga, 
                               div_agrid=div_agrid, rep_cycle=rep_cycle, 
                               cxpb=cxpb, mutpb=mutpb, 
                               toolbox=toolbox, stats=mstats, ruta=ruta,
                               checkpoint = checkpoint,
                               )
    
    #%%
    if len(E)>2:
        f = np.array([ind.fitness.values for ind in E])
        fig, ax = plt.subplots()
        ax.scatter(f[:,0], f[:,1])
        ax.set_xlabel("$1-DSC_{mean}$")
        ax.set_ylabel("$|\Theta|$")
        # ax.set_title("Pareto Front")
        plt.savefig(ruta+"/PF.png", dpi=600)
    else:
        print("No non-dominated solutions found")
    
    #%%%Save execution
    cp = dict(
        SZ_E=sz_e,
        SZ_M=sz_m,
        MNR_p=mnr_p,
        SZ_UP=sz_up,
        ITS_uGA=its_uga,
        ITS=its,
        DIV_AGRID=div_agrid,
        CXPB=cxpb,
        MUTPB=mutpb,
        REP_CYCLE=rep_cycle,
        
        E=E,
        M=M,
        
        LOG = g_log,
         
        time = g_log.select("time")[-1],
        RNDSTATE=random.getstate()
        )
    
    with open(ruta + '/'+ foldername + ".pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)
    
    # #%%Save statistics
    # """Show and save Statics as .png and .csv"""
    # save_statics(log, ruta)
    # show_statics(log, ruta)
    
    # #%%Best individual
    # """Plot Best individual"""
    # best=tools.selBest(pop,1)[0]
    
    # #%%Evo Details
    # """Save details about evolutionary process"""
    # saveEvolutionaryDetails(evolutionary_parameters, best,
    #                         log.select("nevals"), log.select("time"),
    #                         filename=ruta+'/evolution_details.txt')
    
    # #%%Function frequency
    # functionAnalysis(pop,10,pset,ruta)
    
    # #%%Train again the best architecture for reliable compatarion with U-Net
    # best_model=toolbox.clone(best)
    # model=toolbox.make_model(best_model, in_channels=in_channels)
    
    # """Train, val and test loaders"""
    # train_loader, _= dloaders.get_train_loader()
    # val_loader, _  = dloaders.get_val_loader()
    # test_loader, _ = dloaders.get_test_loader()
    
    # """Optimizer"""
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # """Epocas de re-entrenamiento"""
    # nepochs=5
    
    # """Device"""
    # device='cuda:0'
    
    # """For save images or model"""
    # save_model_flag=True
    # save_images_flag=True
    # load_model=False
    
    # #%%Train, validate and test
    # """Train and valid model"""
    # train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
    #     model, train_loader, val_loader,
    #     nepochs, optimizer, lossfn,
    #     device, load_model, save_model_flag,
    #     ruta=ruta, verbose=True
    #     )
    
    # """Test model"""
    # dices, ious, hds = test(test_loader, model, lossfn,
    #                         save_imgs=save_images_flag, ruta=ruta, device=device, verbose=True)
    
    # #%%Attributes assigned
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
    
    # save_execution(ruta, foldername+'.pkl', pop, log, cache, best_model)
    
    # del model
    
    # return log, pop, best_model
    return M, E, g_log

#%%DataFrame for Statistical Test
def make_table(results):
    inds = [str(b[2]) for b in results]
    fitness=[float(b[2].fitness.values[0]) for b in results]
    dices=[b[2].dice for b in results]
    params=[b[2].params for b in results]
    dict={"Arbol":inds, "Fitness":fitness, "Dice":dices, "Params":params}
    daf=pd.DataFrame.from_dict(dict)
    daf.to_csv('/scratch/202201016n/corridas/'+'observaciones.csv')
    

#%% N-Corridas
def n_runs(rango=(1,2)):
    name='NASGP-Net'
    results=[]
    for i in range(rango[0], rango[1]):
        print('Run', i)
        log, pop, best = GPMain(name+str(i))
        print(best, best.fitness, best.dice, best.params)
        results.append([log, pop, best])
    return results
    

if __name__=='__main__':
    # mp.set_start_method('forkserver')
    # log, pop, best = GPMain('MONASGPNet0', 0)
    M, E, log = GPMain('MONASGPNetT', 1)
    
#    results=n_runs(rango=(4,5))
#    make_table(results)
