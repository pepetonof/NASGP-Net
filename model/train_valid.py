# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:57:34 2021

@author: josef
"""
from tqdm import tqdm
import torch
from collections import defaultdict
import torchvision.transforms.functional as TF

from model.utils import load_model, save_model
from model.loss_f import dice_loss

class MetricMonitor:
    def __init__(self, float_precision=3):
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
        
        dice = dice_loss(predictions, targets)
        dice = 1-dice

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # loss acumulated
        running_loss+=loss*train_loader.batch_size
        
        # dice acumulated
        running_dice+=dice*train_loader.batch_size
        
        del images, targets, predictions
        torch.cuda.empty_cache()
        
    loss_out=running_loss/len(train_loader)
    dice_out=running_dice/len(train_loader)
    
   
    if verbose:
        metric_monitor.update("Loss", loss_out.item())
        stream.set_description(
            "Epoch: {epoch}. Train.\t{metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
        print("Epoch: {epoch}. Train.\t{metric_monitor}".format(epoch=epoch, metric_monitor=loss_out.item()))
    
    return loss_out.detach().item(), dice_out.detach().item()

def validate(val_loader, model, loss_fn, epoch, DEVICE, verbose=False):
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
            
            dice = dice_loss(predictions, targets)
            dice = 1-dice
            
            # loss acumulated
            running_loss+=loss*val_loader.batch_size
            
            # dice acumulated
            running_dice+=dice*val_loader.batch_size
            
            
            del images, targets, predictions
            torch.cuda.empty_cache()
            
    loss_out=running_loss/len(val_loader)
    dice_out=running_dice/len(val_loader)
    
    if verbose:
        metric_monitor.update("Loss", loss_out.item())
        stream.set_description(
            "Epoch: {epoch}. Validation.\t{metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
        print("Epoch: {epoch}. Validation.\t{metric_monitor}".format(epoch=epoch, metric_monitor=loss_out.item()))
    
    model.train()
    
    return loss_out.detach().item(), dice_out.detach().item()

def train_and_validate(model, train_loader, val_loader, 
                       num_epochs, optimizer, loss_fn, 
                       DEVICE, LOAD_MODEL, model_save,
                       ruta='results_exp/', verbose=False):
    """Set the model to device"""
    model = model.to(DEVICE)
    
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
