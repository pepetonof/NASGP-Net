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
from model.predict import imgs, overlay_imgs, set_title, save_grid
import numpy as np
import os
import matplotlib.pyplot as plt

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

#%%Early Stopping Function taken from:
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch/71999355#71999355
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(train_loader, model,
          optimizer, loss_fn, metrics, scaler, 
          fold, epoch, device, verbose=False):
    
    model = model.to(device)
    metric_monitor=MetricMonitor()
    model.train()
    
    if verbose:
        stream = tqdm(train_loader)
    else:
        stream = train_loader
    
    # dices = []
    """Metric to compara and loss"""
    metricsTrain = {}
    for m in metrics:
        key=str(type(m)).strip('>').strip("'").split('.')[-1]
        metricsTrain[key]=[]
        if key == 'HDMetric':
            metricsTrain[key+'95']=[]
    #Loss       
    losses = []
    
    for batch_idx, (images, targets) in enumerate(stream, start=1):
        # print('TrainShape', images.shape, targets.shape)
        images = images.to(device=device)
        targets = targets.long().to(device=device)
        
        # print(images.shape, targets.shape)
        
        # forward
        # with torch.cuda.amp.autocast():
        predictions = model(images)
        pred_soft = F.softmax(predictions, dim=1)
    
        #Resize to match espatial information
        if predictions.shape[2:]!=targets.shape[1:]:
            predictions = TF.resize(pred_soft, size=targets.shape[1:])
    
        #Compute loss
        loss = loss_fn(pred_soft, targets)
        losses.append(loss.cpu().detach().item())
        
        #Compute metrics
        """Metrics for evaluation"""
        # with torch.no_grad():
        for m in metrics:
            # print(pred_soft.device, targets.device)
            key=str(type(m)).strip('>').strip("'").split('.')[-1]
            score = m(pred_soft, targets)
            if key == 'HDMetric':  
                metricsTrain[key].append(score[0])
                metricsTrain[key+'95'].append(score[1])
            else:
                metricsTrain[key].append(score)

        if verbose:
            """Monitor Dice for Test Set Validation"""
            dice = metricsTrain['DiceMetric'][-1]
            # print('VERBOSE', dice, dice.item())
            metric_monitor.update("dice", dice)#loss.item())
            if fold!=None:
                # print()
                stream.set_description("Training-{epoch} Fold-{fold} \t\t\t{metric_monitor}".format(epoch=epoch, 
                                                                                                    fold=fold,                                                                             metric_monitor=metric_monitor))
            else:
                # print()
                stream.set_description("Training-{epoch} \t\t{metric_monitor}".format(epoch=epoch, 
                                                                                      metric_monitor=metric_monitor))
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del images, targets, predictions
        torch.cuda.empty_cache()

    model = model.cpu()

    return np.array(losses), metricsTrain#loss_out.detach().item(), dices# dice_out

def validate(val_loader, model, 
             loss_fn, metrics, 
             fold, epoch,
             save_imgs=False, ruta="saved_images/", 
             device="cuda", verbose=False): 
    model = model.to(device)
    
    """Verbose"""
    metric_monitor=MetricMonitor()
    model.eval()
    if verbose:
        stream = tqdm(val_loader)
    else:
        stream = val_loader
    
    """Metric to compara and loss"""
    metricsVal = {}
    for m in metrics:
        key=str(type(m)).strip('>').strip("'").split('.')[-1]
        metricsVal[key]=[]
        if key == 'HDMetric':
            metricsVal[key+'95']=[]
    
    losses = []
    
    """For save  three best predictions"""
    objs=[]
    
    """forward"""
    with torch.no_grad():
        for idx, (x, y) in enumerate(stream, start=1):
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
        
            #Takes each region of interest
            pred_soft = F.softmax(preds, dim=1)
            
            if pred_soft.shape[1:]!=y.shape[1:]:
                pred_soft = TF.resize(pred_soft, size=y.shape[1:])
            
            # print(metrics, idx)
            """Metrics for evaluation"""
            for m in metrics:
                key=str(type(m)).strip('>').strip("'").split('.')[-1]
                score = m(pred_soft, y)
                if key == 'HDMetric':  
                    metricsVal[key].append(score[0])
                    metricsVal[key+'95'].append(score[1])
                else:
                    metricsVal[key].append(score)
            
            ##Loss function
            loss=loss_fn(pred_soft, y)
            losses.append(loss.cpu().detach().item())
            
            if save_imgs:
                # print('Saving')
                # filename = val_loader.dataset[]
                filename_img = val_loader.dataset.image_dir[idx-1].name.split('.')[0]
                # filename_mask = val_loader.dataset.mask_dir[idx-1].name.split('.')[0]
                path_imgs = f"{ruta}/validate_segmentation/fold_{fold}" if fold!=None else f"{ruta}/validate_segmentation"
                if not os.path.exists(path_imgs):
                    os.makedirs(path_imgs)
                
                sem_classes=['__background__'] + ["roi" + str(i) for i in range(1,preds.shape[1])]
                sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)} 
                
                y_roi = []
                pred_roi = []
                #For each class takes the predicted and groundtruth mask
                #Ignore background
                for key in list(sem_class_to_idx.keys())[1:]:
                    # class_dim = sem_class_to_idx[key]
                    pbool = pred_soft.argmax(1) == sem_class_to_idx[key]
                    ybool = y ==sem_class_to_idx[key]
                    
                    pred_roi.append(pbool.unsqueeze(dim=1))
                    y_roi.append(ybool.unsqueeze(dim=1))
                    
                pred_roi = torch.cat(pred_roi, dim=1)
                y_roi = torch.cat(y_roi, dim=1)
                # print('X Img', x.shape)
                # print("Pred_ROI", pred_roi.shape, torch.unique(pred_roi))
                # print("Y_ROI", y_roi.shape, torch.unique(y_roi))
                
                #Obtain original images with masks and predictions on top
                over_masks, over_preds = overlay_imgs(x, y_roi, pred_roi)
                
                #Save original image
                if epoch==1:
                    torchvision.utils.save_image(x, f"{path_imgs}/{filename_img}.png")
                    # print(x.max(), x.shape, x.dtype)
                    over_masks = (over_masks.float())/255.00
                    # print(over_masks.max(), over_masks.shape, over_masks.dtype)
                    title = f"{path_imgs}/{filename_img}_mask_F{fold}.png" if fold!=None else f"{path_imgs}/{filename_img}_mask.png"
                    torchvision.utils.save_image(over_masks, title)
                
                # print(over_preds.shape, over_masks.shape)
                
                #Set dice index to the over_preds
                dice = metricsVal['DiceMetric'][-1]
                over_preds=set_title(over_preds, 'Dice='+str(round(dice,3)))
                
                #For save with torch.utils.save_image()
                over_preds = (over_preds.float())/255.00
                # print(over_preds.max(), over_preds.shape, over_preds.dtype)
                # print(over_preds.shape, over_masks.shape)
                
                title = f"{path_imgs}/{filename_img}_pred_E{epoch}F{fold}_dice={round(dice,3)}.png" if fold else f"{path_imgs}/{filename_img}_pred_E{epoch}_dice={round(dice,3)}.png"
                torchvision.utils.save_image(over_preds, title)
            
                #Save grid of the three best/worst individuals
                # iou = metricsVal['IoUMetric'][-1]
                # hd = metricsVal['HDMetric95'][-1]
                obj_best=imgs(x, over_masks, over_preds, dice)#, iou, hd)
                objs.append(obj_best)
            
            if verbose:
                dice = metricsVal['DiceMetric'][-1]
                metric_monitor.update("dice", dice)
                if fold!=None:
                    stream.set_description(
                        "Validation-{epoch} Fold-{fold}\t\t\t{metric_monitor}".format(epoch=epoch,
                                                                                      fold=fold,
                                                                                      metric_monitor=metric_monitor))
                else:
                    stream.set_description(
                        "Validation-{epoch} \t\t{metric_monitor}".format(epoch=epoch,
                                                                         metric_monitor=metric_monitor))
            #For saving memory
            del x, y, preds
            torch.cuda.empty_cache()
    
    #Save grid of the three best/worst individual
    if save_imgs:
        title = f"{path_imgs}/best_segmentations_E{epoch}F{fold}.png" if fold!=None else f"{path_imgs}/best_segmentations_E{epoch}.png"
        save_grid(objs, 3, 'best', title)
        title = f"{path_imgs}/worst_segmentations_E{epoch}F{fold}.png" if fold!=None else f"{path_imgs}/worst_segmentations_E{epoch}.png"
        save_grid(objs, 3, 'worst', title)
    
    model.train()
    model=model.cpu()
    
    return np.array(losses), metricsVal


#%%Train and validate
def train_and_validate(model, train_loader, val_loader,
                       num_epochs, tolerance,
                       optimizer, loss_fn, metrics,
                       device, save_model=False, save_images=False,
                       ruta='results_exp/', verbose=False, fold=None):

    """For monitoring and graph train and valid loss""" 
    train_loss, valid_loss = [], []
    
    """For monitoring train and valid dice for fitness computation""" 
    train_dice, valid_dice = [], []
    
    """Scaler"""
    scaler = torch.cuda.amp.GradScaler()
    
    "Early stopping"
    early_stopper = EarlyStopper(patience=tolerance, min_delta=0)
    
    for epoch in range(1,num_epochs+1):
        losses_t, metricsTrain = train(train_loader, model, 
                                  optimizer, loss_fn, metrics[:1], scaler, 
                                  fold, epoch, device, verbose)
        
        train_loss.append(np.mean(losses_t))
        train_dice.append(np.mean(metricsTrain["DiceMetric"]))
        
        # print('Trained')
        
        losses_v, metricsVal = validate(val_loader, model, loss_fn, metrics[:1],
                                         fold, epoch, 
                                         save_imgs=save_images, ruta=ruta, 
                                         device=device, verbose=verbose)
        
        valid_loss.append(np.mean(losses_v))
        valid_dice.append(np.mean(metricsVal["DiceMetric"]))
        
        if early_stopper.early_stop(np.mean(losses_v)):
            if verbose:
                print(f"... stopped in epoch: {epoch}")
            break
    
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    
    train_dice = np.array(train_dice)
    valid_dice = np.array(valid_dice)

    if save_model:
        path_models = f"{ruta}/model/fold_{fold}" if fold!=None else f"{ruta}/model"
        if not os.path.exists(path_models):
            os.makedirs(path_models)
        title = f"{path_models}/model-fold{fold}.pth.tar" if fold!=None else f"{path_models}/model.pth.tar"
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),}
        torch.save(checkpoint, title)
    
    if save_images:
        # print('ruta', ruta)
        path_imgs=f"{ruta}/dice_loss/fold_{fold}" if fold!=None else f"{ruta}/dice_loss"
        # print('path_images', path_imgs)
        if not os.path.exists(path_imgs):
            os.makedirs(path_imgs)
        #Also, save the train_loss, valid_loss, train_dice, valid_dice
        #Train loss and valid loss
        epochs=[i for i in range(len(train_loss))]
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(epochs, train_loss, "b-", label="Train loss")
        line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid loss")
        ax1.set_xlabel("Epoch")
        ax1.set_xticks(np.arange(0, len(epochs), 2))
        ax1.set_ylabel("Loss")
        lns=line1+line2
        labs = [l.get_label() for l in lns]
        ax1.legend(line1+line2, labs, loc="center right")
        
        plt.close(fig)
        plt.show()
        title = f"{path_imgs}/model-fold{fold}-loss.png" if fold!=None else f"{path_imgs}/model-loss.png"
        fig.savefig(title,dpi=600)
        
        #Train dice and valid dice
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(epochs, train_dice, "b-", label="Train Dice")
        line2 = ax1.plot(epochs, valid_dice, "r-", label="Valid Dice")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Dice")
        ax1.set_xticks(np.arange(0, len(epochs), 2))
        lns=line1+line2
        labs = [l.get_label() for l in lns]
        ax1.legend(line1+line2, labs, loc="center right")
        
        plt.close(fig)
        plt.show()
        title = f"{path_imgs}/model-fold{fold}-dice.png" if fold!=None else f"{path_imgs}/model-dice.png"
        fig.savefig(title, dpi=600)
    
    lossAndDice = {
        "train_loss":train_loss,
        "valid_loss":valid_loss,
        "train_dice":train_dice,
        "valid_dice":valid_dice}
    
    return lossAndDice, metricsVal