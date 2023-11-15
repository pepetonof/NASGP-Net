# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:57:34 2021

@author: josef
"""
from tqdm import tqdm
import torch
from collections import defaultdict
import torchvision.transforms.functional as TF
# import torchvision.transforms.functionalJ as TF

import torch.nn.functional as F
import torchvision

from model.utils import load_model, save_model
# from model.loss_f import dice_loss
# from model.metrics import dice_score

from model.metrics import dice_score, iou_score
from model.loss_f import hdistance_loss
from model.predict import imgs, overlay_imgs, set_title, save_grid
import numpy as np

class MetricMonitor:
    def __init__(self, float_precision=6):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def train(train_loader, model, optimizer, loss_fn, scaler, epoch, DEVICE, verbose=False):
    model = model.to(DEVICE)
    
    metric_monitor=MetricMonitor()
    model.train()
    if verbose:
        stream = tqdm(train_loader)
    else:
        stream = train_loader
    running_loss=0.0
    running_dice=0.0
    
    for batch_idx, (images, targets) in enumerate(stream, start=1):
        #print('Train', images.shape, targets.shape)
        images = images.to(device=DEVICE)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        
        # forward
        # with torch.cuda.amp.autocast():
        predictions = model(images)
        
        if predictions.shape[2:]!=targets.shape[1:]:
            predictions = TF.resize(predictions, size=targets.shape[1:])
        #print('Train', predictions.shape)

        loss = loss_fn(predictions, targets)
        
        dice = dice_score(predictions, targets)
        # dice = 1-dice
        
        
        if verbose:
            """Monitor Dice for Test Set Validation"""
            metric_monitor.update("dice", dice)#loss.item())
            stream.set_description("Training-{epoch} \t\t{metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
        

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # loss acumulated
        running_loss+=loss#*train_loader.batch_size
        
        # dice acumulated
        running_dice+=dice#*train_loader.batch_size
        
        del images, targets, predictions
        torch.cuda.empty_cache()
        
    loss_out=running_loss/len(train_loader)
    dice_out=running_dice/len(train_loader)
    
   
    # if verbose:
    #     metric_monitor.update("Loss", loss_out.item())
    #     stream.set_description(
    #         "Epoch: {epoch}. Train.\t{metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
    #         )
    #     print("Epoch: {epoch}. Train.\t{metric_monitor}".format(epoch=epoch, metric_monitor=loss_out.item()))
    
    model = model.cpu()
    
    return loss_out.detach().item(), dice_out

def validate2(test_loader, model, loss_fn,
              save_imgs=False, ruta="saved_images/", device="cuda", verbose=False):
    model = model.to(device)
    
    metric_monitor=MetricMonitor()
    model.eval()
    if verbose:
        stream = tqdm(test_loader)
    else:
        stream = test_loader
    
    """Metrics to compare"""
    dices= []
    ious = []
    hds  = []
    hds95= []
    
    losses = []
    # nsds = []
    
    """For save  three best predictions"""
    objs=[]
    
    # if save_imgs:
    #     print('=> Saving images...')
    
    with torch.no_grad():
    
        for idx, (x, y) in enumerate(stream, start=1):
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
        
            #Takes each region of interest
            # pbool = F.softmax(preds, dim=1)
            pred_soft = F.softmax(preds, dim=1)
            
            if pred_soft.shape[1:]!=y.shape[1:]:
                pred_soft = TF.resize(pred_soft, size=y.shape[1:])
            
            # sem_classes=['__background__', 'placenta']
            sem_classes=['__background__'] + ["roi" + str(i) for i in range(1,preds.shape[1])]
            sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)} 
            
            # print("\nMask-", torch.unique(y), y.shape)
            # print("PredShape", pred_soft.shape)
            # print("Pred-",)
            
            #For each class
            y_roi = []
            pred_roi = []
            for key in list(sem_class_to_idx.keys())[1:]:
                # class_dim = sem_class_to_idx[key]
                pbool = pred_soft.argmax(1) == sem_class_to_idx[key]
                ybool = y ==sem_class_to_idx[key]
                
                # print("Pbool Shape:\n", pbool.shape, key)
                # print("Ybool Shape:\n", ybool.shape, key)
                
                pred_roi.append(pbool.unsqueeze(dim=1))
                y_roi.append(ybool.unsqueeze(dim=1))
                
            pred_roi = torch.cat(pred_roi, dim=1)
            y_roi = torch.cat(y_roi, dim=1)
            # print("Pred_ROI", pred_roi.shape, torch.unique(pred_roi))
            # print("Y_ROI", y_roi.shape, torch.unique(y_roi))
            
            # class_dim=1
            # pbool = pbool.argmax(class_dim) == sem_class_to_idx['placenta']
            
            # if pbool.shape[1:]!=y.shape[1:]:
            #     pbool = TF.resize(pbool, size=y.shape[1:])
            
            """Metrics for evaluation"""
            #Dice coefficient
            # dice = dice_loss(preds, y)
            # dice = 1-dice
            # dice = dice.detach().item()
            dice = dice_score(preds, y)
            # print("\nDiceScore:\t", dice)
            dices.append(dice) #Append to dices
            
            # """Monitor Loss for Test Set Validation"""
            # metric_monitor2.update("Loss:", loss)
            # stream.set_description("Testing. {metric_monitor}".format(metric_monitor=metric_monitor2))
        
            #IoU
            # iou = IoU_loss(preds, y)
            # iou = 1-iou
            # iou =iou.detach().item()
            iou = iou_score(preds, y)
            ious.append(iou)#Append to ious
            
            #Hausdorff Distance
            hd, hd95 = hdistance_loss(preds, y)
            hds.append(hd)
            hds95.append(hd95)
            
            ##Loss function
            loss=loss_fn(preds, y)
            losses.append(loss.detach().item())
            
            #NSD
            # nsd = nsdf(preds, y)
            # nsds.append(nsd)
            
            if save_imgs:
                #Save each image
                #Save original image
                torchvision.utils.save_image(x, f"{ruta}/{idx}_in.png")
                #Obtain original images with masks and predictions on top
                # over_masks, over_preds = overlay_imgs(x, y, pbool)
                over_masks, over_preds = overlay_imgs(x, y_roi, pred_roi)
                
                #Set dice index to the over_preds
                # over_preds=set_title(over_preds, 'Dice='+str(round(dice,6)))
                
                #For save with torch.utils.save_image()
                over_masks = (over_masks.float())/255.00
                over_preds = (over_preds.float())/255.00
                # print(preds.shape)
                # print(over_masks.shape,over_preds.shape)
                
                torchvision.utils.save_image(over_masks, f"{ruta}/{idx}_overmask.png")
                torchvision.utils.save_image(over_preds, f"{ruta}/{idx}_overprd_dice={round(dice,3)}.png")
            
                #Save grid of the three best/worst individuals
                obj_best=imgs(x, over_masks, over_preds, dice, iou, hd)
                objs.append(obj_best)
            
            if verbose:
                """Monitor Dice for Test Set Validation"""
                metric_monitor.update("dice", dice)#loss.item())
                stream.set_description("Validation. \t\t{metric_monitor}".format(metric_monitor=metric_monitor))

            #For saving memory
            del x, y, preds
            torch.cuda.empty_cache()
        
    # if verbose:
    #     """Monitor Dice for Test Set Validation"""
    #     metric_monitor.update("dice", dice)#loss.item())
    #     print(metric_monitor)
    #     stream.set_description("Testing. \t\t{metric_monitor}".format(metric_monitor=dice))
    
    #Save grid of the three best/worst individual
    if save_imgs:
        save_grid(objs, 3, 'best', ruta)
        save_grid(objs, 3, 'worst', ruta)
    
    if verbose:        
        print(f"Got Dice score mean:\t {np.mean(dices):.8f}")
        print(f"Got Dice score max:\t {np.max(dices):.8f}")
        print(f"Got Dice score min:\t {np.min(dices):.8f}")
        print(f"Got Dice score std:\t {np.std(dices):.8f}")
    
    model.train()
    model=model.cpu()
    
    return dices, ious, hds, hds95, losses

def validate(val_loader, model, loss_fn, epoch, DEVICE, verbose=False):
    model = model.to(DEVICE)
    
    metric_monitor=MetricMonitor()
    model.eval()
    if verbose:
        stream = tqdm(val_loader)
    else:
        stream = val_loader
    running_loss=0.0
    running_dice=0.0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(stream, start=1):
            images = images.to(device=DEVICE)
            targets = targets.long().to(device=DEVICE)
            predictions = model(images)
            
            if predictions.shape[2:]!=targets.shape[1:]:
                predictions = TF.resize(predictions, size=targets.shape[1:])
            
            loss=loss_fn(predictions, targets)
            
            dice = dice_score(predictions, targets)
            # dice = 1-dice
            
            # loss acumulated
            running_loss+=loss#*val_loader#.batch_size
            
            # dice acumulated
            running_dice+=dice#*val_loader#.batch_size
            
            if verbose:
                metric_monitor.update("dice", dice)
                stream.set_description(
                    "Validation-{epoch}\t\t{metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
                    )
            
            
            del images, targets, predictions
            torch.cuda.empty_cache()
            
            
            
    loss_out=running_loss/len(val_loader)
    dice_out=running_dice/len(val_loader)
    
    
    
    model.train()
    
    model = model.cpu()
    
    return loss_out.detach().item(), dice_out#.detach().item()

def train_and_validate(model, train_loader, val_loader, 
                       num_epochs, optimizer, loss_fn, 
                       DEVICE, LOAD_MODEL, model_save,
                       ruta='results_exp/', verbose=False):
    # """Set the model to device"""
    # model = model.to(DEVICE)
    
    """Load model"""
    if LOAD_MODEL:
        load_model(torch.load(ruta+"/my_checkpoint.pth.tar"), model)
    
    """For monitoring and graph train and valid loss""" 
    train_loss, valid_loss = [], []
    
    """For monitoring and graph train and valid dice""" 
    train_dice, valid_dice = [], []
    
    """Scaler"""
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(1,num_epochs+1):
        losst, dicet = train(train_loader, model, 
                      optimizer, loss_fn, scaler, epoch, DEVICE, verbose)
        train_loss.append(losst)
        train_dice.append(dicet)
        
        lossv, dicev = validate(val_loader, model, loss_fn, epoch, DEVICE, verbose)
        valid_loss.append(lossv)
        valid_dice.append(dicev)
        
    if model_save:
        save_model(model, optimizer, ruta)
    
    return train_loss, valid_loss, train_dice, valid_dice