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

from data.dataloader import loaders
from torchinfo import summary
from sklearn.model_selection import KFold

import pickle
import os

def make_model(ind, in_channels, out_channels, pset):
    """Compile function"""
    func = compile(expr=ind, pset=pset)
    """Init module: empty sequential module"""
    init_module=[nn.ModuleList(), in_channels]
    """Output of the first block"""
    first_block=func(init_module)
    model=BackBone(first_block, out_channels)
    return model
    
def evaluation_cv(ind, 
                  nepochs, 
                  tolerance,
                  lossfn,
                  metrics,
                  lr, 
                  dataset_type,
                  no_classes_msk,
                  in_channels, 
                  out_channels,
                  batch_size,
                  image_height,
                  image_width,
                  max_params, w, 
                  k_folds,
                  # split,
                  train_set, 
                  valid_set, 
                  test_set,
                  pset, 
                  device,
                  ruta,
                  verbose_train,
                  save_model,
                  save_images,
                  save_datafolds,
                  limit = 300000000
                  ):
    
    #Generate model
    model = make_model(ind, in_channels, out_channels, pset)
    
    #Evaluate no of parameters
    complexity, params = evaluate_NoParameters(model, max_params)
    
    #Check the max number of allowed parameters, 
    #if does not fit in the RAM, just assign an bad fit
    if params>=limit:
        cv_dices = [0.0]*k_folds
        cv_ious = [0.0]*k_folds
        cv_hds = [10.0]*k_folds
        cv_hds95 = [9.5]*k_folds
        cv_nsds = [10.0]*k_folds
     
    #else, if fits in the RAM continue with k-fold    
    else:
        #Cross Validation Information
        cv_dices=[]
        cv_ious=[]
        cv_hds=[]
        cv_hds95=[]
        cv_nsds=[]
        # cv_add_train=[]
        
        if type(k_folds)==int:
            #Evaluate Segmentation Metrics
            kfold = KFold(n_splits=k_folds, shuffle=False)
        
            #Merge train and valid. Leave test
            data_fold = train_set["images"]+valid_set["images"]
            mask_fold = train_set["masks"]+valid_set["masks"]
        
            # K-fold Cross Validation model evaluation
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(data_fold),1):
                #Folds
                train_set = dict(images = list(map(data_fold.__getitem__, train_idx)), 
                                 masks = list(map(mask_fold.__getitem__, train_idx)))
                valid_set = dict(images = list(map(data_fold.__getitem__, valid_idx)), 
                                 masks = list(map(mask_fold.__getitem__, valid_idx)))
                
                # print('Fold No Split', fold, type(train_set), type(valid_set), type(test_set))
                #Dataset depending on data
                dloaders = loaders(train_set, valid_set, test_set, batch_size=batch_size, 
                                   image_height=image_height, image_width=image_width,
                                   dataset_type=dataset_type, no_classes_msk=no_classes_msk,
                                   )
                
                #Evaluate Segentation
                metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
                                                                              lr, dloaders,
                                                                              device, ruta, verbose_train,
                                                                              save_model, save_images, fold=fold)
                
                #Save information about fold
                if save_datafolds:
                    path_data = f"{ruta}/data/fold_{fold}"
                    if not os.path.exists(path_data):
                        os.makedirs(path_data)
                    with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
                        pickle.dump(metricsTest, cp_file)
                    with open(f"{path_data}/metricsValfold_{fold}.pkl", "wb") as cp_file:
                        pickle.dump(metricsVal, cp_file)
                    with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
                        pickle.dump(lossAndDice, cp_file)
                    
                #Aggregate CV information
                cv_dices.append(np.mean(metricsTest["DiceMetric"]))
                cv_ious.append(np.mean(metricsTest["IoUMetric"]))
                cv_hds.append(np.mean(metricsTest["HDMetric"]))
                cv_hds95.append(np.mean(metricsTest["HDMetric95"]))
                cv_nsds.append(np.mean(metricsTest["NSDMetric"]))
                # cv_add_train.append(add_train)
                
                #Print
                # if verbose_train:
                #     print('DSC_mean for fold %d: %f' % (fold, cv_dices[-1]))
                #     print('Iou_mean for fold %d: %f' % (fold, cv_ious[-1]))
                #     print('HDS_mean for fold %d: %f' % (fold, cv_hds[-1]))
                #     print('HDS95_mean for fold %d: %f' % (fold, cv_hds95[-1]))
                #     print('NSD_mean for fold %d: %f' % (fold, cv_nsds[-1]))
                
                #Generate model again to reinitialize the weights in each fold
                # model.initialize_weigths()
                model = make_model(ind, in_channels, out_channels, pset)
            
        elif k_folds==False:#Ignore fold due the folds are already computed
            #Evaluate pre computed folds. The paths to files are stored in train_set, test_set, valid_set as a list of dictionaries similar to those used when split is false
            # a=len(train_set)
            # print(a)
            for fold in range(1, len(train_set)+1, 1):
                #Split train to obtain valid set:
                train_set_fold = train_set[fold-1]
                valid_set_fold = valid_set[fold-1]
                test_set_fold = test_set[fold-1]
                
                # print('Fold Split', fold, type(train_set_fold), type(valid_set_fold), type(test_set_fold))
                
                #Dataset depending on data
                dloaders = loaders(train_set_fold, valid_set_fold, test_set_fold, batch_size=batch_size, 
                                   image_height=image_height, image_width=image_width,
                                   dataset_type=dataset_type, no_classes_msk=no_classes_msk,
                                   )
                # print('ModelDevice', next(model.parameters()).device)
                #Evaluate Segentation
                metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
                                                                              lr, dloaders,
                                                                              device, ruta, verbose_train,
                                                                              save_model, save_images, fold=fold)
                #Save information about fold
                if save_datafolds:
                    path_data = f"{ruta}/data/fold_{fold}"
                    if not os.path.exists(path_data):
                        os.makedirs(path_data)
                    with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
                        pickle.dump(metricsTest, cp_file)
                    with open(f"{path_data}/metricsValfold_{fold}.pkl", "wb") as cp_file:
                        pickle.dump(metricsVal, cp_file)
                    with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
                        pickle.dump(lossAndDice, cp_file)
                    
                #Aggregate CV information
                cv_dices.append(np.mean(metricsTest["DiceMetric"]))
                cv_ious.append(np.mean(metricsTest["IoUMetric"]))
                cv_hds.append(np.mean(metricsTest["HDMetric"]))
                cv_hds95.append(np.mean(metricsTest["HDMetric95"]))
                cv_nsds.append(np.mean(metricsTest["NSDMetric"]))
                # cv_add_train.append(add_train)
                
                #Print
                # if verbose_train:
                #     print('DSC_mean for fold %d: %f' % (fold, cv_dices[-1]))
                #     print('Iou_mean for fold %d: %f' % (fold, cv_ious[-1]))
                #     print('HDS_mean for fold %d: %f' % (fold, cv_hds[-1]))
                #     print('HDS95_mean for fold %d: %f' % (fold, cv_hds95[-1]))
                #     print('NSD_mean for fold %d: %f' % (fold, cv_nsds[-1]))
                
                #Generate model again to reinitialize the weights in each fold
                model = make_model(ind, in_channels, out_channels, pset)
                # model.initialize_weigths()

    #Fitness as lienar combination of mean dice and the number of parameters
    fit = (1 - w)*np.mean(cv_dices) + w*complexity
    
    #Fitness as minimization
    # fit = (1-w)*np.mean(cv_dices_loss) + w*(params/max_params)
    
    # alpha = 0.25
    # beta = 0.25
    # DSC_loss_train = alpha*np.mean(1 - np.array(cv_dices_t))
    # DSC_loss_val   = np.mean(1 - np.array(cv_dices))
    # ADD_train      = beta*np.mean(np.array(cv_add_train))
    
    # f1 = DSC_loss_val + beta*ADD_train + alpha*DSC_loss_train
    # f2 = params/max_params
    # print(f1,f2)
    # fit= (1-w)*f1 + w*f2

    # print(f1, DSC_loss_val, DSC_loss_train, DSC_loss_train*alpha, ADD_train, f2, params, max_params)
    #np.mean(cv_dices_t),
    del model
    return fit, np.mean(cv_dices), np.mean(cv_ious), np.mean(cv_hds), np.mean(cv_hds95), np.mean(cv_nsds), params #hd95, nds
    
def evaluation(ind,
                nepochs,
                tolerance,
                lossfn,
                metrics,
                lr,
                dataset_type,
                no_classes_msk,
                in_channels, 
                out_channels,
                batch_size,
                image_height,
                image_width,
                max_params, w,
                train_set,
                valid_set,
                test_set,
                pset,
                device,
                ruta,
                verbose_train,
                save_model,
                save_images,
                save_data,
                limit=300000000,
                ):
    
    #Generate model
    model = make_model(ind, in_channels, out_channels, pset)
    
    #Evaluate no of parameters
    complexity, params = evaluate_NoParameters(model, max_params)
    
    #Check the max number of allowed parameters, 
    #if does not fit in the RAM, just assign an bad fit
    if params >= limit:
            dice =0.0
            iou = 0.0
            hd = 10
            hd95 = 10*0.95
    #else, if fits in the RAM continue with train and validation
    else:
        
        #Dataloaders from train_set, valid_set, test_set
        dloaders = loaders(train_set, valid_set, test_set, batch_size=batch_size, 
                            image_height=image_height, image_width=image_width, 
                            dataset_type=dataset_type, no_classes_msk=no_classes_msk,
                            )
        
        #Evaluate Segmentation Metrics
        metricsTest, metricsVal, lossAndDice = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics,
                                                                     lr, dloaders, 
                                                                     device, ruta, verbose_train,
                                                                     save_model, save_images, fold=None)
        # Segmentation metrics: Mean value on the test set
        dice = np.mean(metricsTest["DiceMetric"])
        iou  = np.mean(metricsTest["IoUMetric"])
        hd   = np.mean(metricsTest["HDMetric"])
        hd95 = np.mean(metricsTest["HDMetric95"])
        nsd  = np.mean(metricsTest["NSDMetric"])
        
        # print(metricsTest)
        # print(metricsVal)
        
        if save_data:
            path_data = f"{ruta}/data/"
            if not os.path.exists(path_data):
                os.makedirs(path_data)
            with open(f"{path_data}/metricsTest.pkl", "wb") as cp_file:
                pickle.dump(metricsTest, cp_file)
            with open(f"{path_data}/metricsVal.pkl", "wb") as cp_file:
                pickle.dump(metricsVal, cp_file)
            with open(f"{path_data}/lossAndDice.pkl", "wb") as cp_file:
                pickle.dump(lossAndDice, cp_file)
            d={}
            d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
                     "Width":dloaders.IMAGE_WIDTH, 
                     "Train_Size":len(dloaders.TRAIN_IMG_DIR),
                     "Valid_Size":len(dloaders.VAL_IMG_DIR),
                     "Test_Size": len(dloaders.TEST_IMG_DIR),}
            d.update(d_aux)
            d_aux = {"Best":ind, "Best_Fitness":ind.fitness.values[0], 
                   
                    "DiceMean": dice, "DiceMedian":np.median(metricsTest["DiceMetric"]), "DiceMax":np.max(metricsTest["DiceMetric"]),
                    "DiceMin": np.min(metricsTest["DiceMetric"]), "DiceStd":np.std(metricsTest["DiceMetric"]),
                   
                    "IoUMean": iou, "IoUMedian":np.median(metricsTest["IoUMetric"]), "IoUMax": np.max(metricsTest["IoUMetric"]),
                    "IoUMin": np.min(metricsTest["IoUMetric"]), "IoUStd": np.std(metricsTest["IoUMetric"]),
                   
                    "HdMean": hd, "HdMedian":np.median(metricsTest["HDMetric"]), "HdMax": np.max(metricsTest["HDMetric"]),
                    "HdMin": np.min(metricsTest["HDMetric"]), "HdStd": np.std(metricsTest["HDMetric"]),
                    
                    "Hd95Mean": hd95, "HD95Median":np.median(metricsTest["HDMetric95"]), "Hd95Max": np.max(metricsTest["HDMetric95"]),
                    "Hd95Min": np.min(metricsTest["HDMetric95"]), "Hd95Std": np.std(metricsTest["HDMetric95"]),
                    
                    "NSDMean": nsd, "NSDMedian":np.median(metricsTest["NSDMetric"]), "NSDMax": np.max(metricsTest["NSDMetric"]),
                    "NSDMin": np.min(metricsTest["NSDMetric"]), "NSDStd": np.std(metricsTest["NSDMetric"]),                    
                    }
            d.update(d_aux)
            model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
            summary_model = str(model_stats)
            d_aux = {"Summary_Model": summary_model}
            d.update(d_aux)
            
            with open(f"{path_data}/data.txt", 'w', encoding="utf-8") as f: 
                for key, value in d.items(): 
                    f.write('%s\n%s\n' % (key, value))

    #Fitness as lienar combination of mean dice and the number of parameters
    fit = (1 - w)*dice + w*complexity
    
    return fit, dice, iou, hd, hd95, nsd, params
        

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
