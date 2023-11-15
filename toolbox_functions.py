# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:25:34 2022

@author: josef
"""
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygraphviz as pgv
import matplotlib.pyplot as plt
# import networkx as nx
from deap.gp import *
import hashlib
# import deap.pg as pg

from model.model import BackBone
from model.train_valid import train_and_validate
from model.predict import test
from objective_functions import evaluate_NoParameters, evaluate_Segmentation

def make_model(ind, in_channels, out_channels, pset):
    """Compile function"""
    func = compile(expr=ind, pset=pset)
    """Init module: empty sequential module"""
    init_module=[nn.ModuleList(), in_channels]
    """Output of the first block"""
    first_block=func(init_module)
    model=BackBone(first_block, out_channels)
    return model
    
def evaluation(ind, nepochs, lossfn, lr,
               max_params, w,
                
               loaders, pset, device, ruta, verbose_train):
    
    # """Make model"""
    in_channels = loaders.IN_CHANNELS
    out_channels = loaders.OUT_CHANNELS
    model = make_model(ind, in_channels, out_channels, pset)
    
    #Evaluate Segmentation Metrics
    metrics_seg, _ = evaluate_Segmentation(model, nepochs, lossfn, lr, loaders, 
                                                 device, ruta, verbose_train)
    
    # Segmentation metrics: Mean value on the test set
    dice = np.mean(metrics_seg["dices"])
    iou  = np.mean(metrics_seg["ious"])
    hd   = np.mean(metrics_seg["hds"])
    
    hd95 = np.mean(metrics_seg["hds95"])
    # nsd  = np.mean(metrics_seg["nsd"])
    
    #Evaluate no of parameters
    complexity, params = evaluate_NoParameters(model, loaders.IN_CHANNELS, max_params, pset)
    
    #Fitness as lienar combination of mean dice and the number of parameters
    fit = (1 - w)*dice + w*complexity
    
    #return fit, dice, params, metrics_seg#, metrics, train_valid, params
    # return fit, metrics_seg, params
    return fit, dice, iou, hd, hd95, params #hd95, nds
    

def evaluationMO(ind, nepochs, lossfn, lr,
                 max_params,
                
                 loaders, pset, device, ruta, verbose_train):
    
    # """Make model"""
    in_channels = loaders.IN_CHANNELS
    model = make_model(ind, in_channels, pset)
    
    #Evaluate no of parameters
    complexity, params = evaluate_NoParameters(model, loaders.IN_CHANNELS, max_params, pset)
    # print(params)
    
    #Exceed on COVID
    #121219529:#680810886:#680810886 #mpool(cat(dCon(conv(mod, 32, 5, 5, 2), 0.7), dCon(sconv(conv(conv(conv(sconv(mod, 32, 5, 7, 1), 16, 7, 3, 2), 32, 7, 5, 2), 32, 7, 7, 1), 8, 5, 7, 1), 0.8)))
    if params<125000000:
    
        #Evaluate Segmentation Metrics
        metrics, train_valid = evaluate_Segmentation(model, nepochs, lossfn, lr, loaders, 
                                                     device, ruta, verbose_train)
        
        #Evaluate segmentation performance. Use the mean dice
        dice=np.mean(metrics["dices"])
    else:
        dice = 0.9#?
    
    # #Fitness as lienar combination of mean dice and the number of parameters
    # fit = (1 - w)*dice + w*complexity
    
    return 1-dice, params, #fit, dice, params#, metrics, train_valid, params
    
def evaluationMP(ind, nepochs, lossfn, lr,
                  max_params, w,
                 
                  loaders, pset, device, ruta, verbose_train):
    
    """Train, val and test loaders"""
    train_loader, _ = loaders.get_train_loader() #loaders.get_train_loader(288, 480)
    val_loader, _  = loaders.get_val_loader()
    test_loader, _ = loaders.get_test_loader()
    

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    """Hyoperparameters for train"""
    LOAD_MODEL = False
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    """Save model and images"""
    save_model=False
    save_images=False
    
    """Train and valid model"""
    train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
        model, train_loader, val_loader,
        nepochs, optimizer, lossfn,
        device, LOAD_MODEL, save_model,
        ruta=ruta, verbose=verbose_train
        )
    
    """Test model"""
    dices, ious, hds = test(test_loader, model, lossfn,
                            save_imgs=save_images, ruta=ruta, device=device, verbose=verbose_train)
    
    metrics={}
    graphs={}
    
    metrics["dices"]=dices
    metrics["ious"]=ious
    metrics["hds"]=hds
    
    graphs["train_loss"]=train_loss
    graphs["valid_loss"]=valid_loss
    graphs["train_dice"]=train_dice
    graphs["valid_dice"]=valid_dice
    
    fitness = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params)
    print('Syntax tree:\t', str(ind), round(fitness,3), round(np.mean(dices),3), params)

    #manager_list.append([num_process, fitness, np.mean(dices), params, train_loss, valid_loss, train_dice, valid_dice, dices, ious, hds])
    # return fitness,
    return fitness, np.mean(dices), params, metrics, graphs

"""Functions fot storage model, train and valid loss, graph as .png and .txt and segmented images"""
def save_ind(ind, ruta, filename='tree'):
    tree=PrimitiveTree(ind)
    nodes, edges, labels = graph(tree)
    g = pgv.AGraph(directed=False)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]   
    
    txt_f=open(ruta + "/"+filename+".txt", "w")
    txt_f.write(str(ind))
    txt_f.write('Fitness:\t'+str(ind.fitness.values))
    txt_f.write('Dice:\t'+str(ind.dice))
    txt_f.write('Params:\t'+str(ind.params))
    txt_f.close()
    
    g.draw(ruta + "/" + filename + '.png')
    return

# """Shows a tree that represents an individual"""
# def plt_ind(ind):
#     tree=PrimitiveTree(ind)
#     nodes, edges, labels = graph(tree)
#     g = nx.Graph()
#     g.add_nodes_from(nodes)
#     g.add_edges_from(edges)
    
#     pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    
#     nx.draw_networkx_nodes(g, pos)
#     nx.draw_networkx_edges(g, pos)
#     nx.draw_networkx_labels(g, pos, labels)
#     plt.axis('off')
#     plt.show()
#     return

"""Shows and save graph of valid an train loss"""
def save_graphtvd(ind, ruta, filename, show=False):
    train_loss=ind.train_dice
    valid_loss=ind.valid_dice
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train dice")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid dice")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)

"""Shows and save graph of valid an train loss"""
def save_graphtv(ind, ruta, filename, show=False):
    train_loss=ind.train_loss
    valid_loss=ind.valid_loss
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train loss")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)

def identifier(ind, length=10):
    string=str(ind)
    return hashlib.sha224(string.encode('utf-8')).hexdigest()[-length:]
