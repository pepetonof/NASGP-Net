# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 02:24:13 2021

@author: josef
"""
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms.functional as TF
# import torchvision.transforms.functionalJ as TF
import matplotlib.pyplot as plt
import copy

from model.loss_f import hdistance_loss
from model.metrics import dice_score, iou_score
from collections import defaultdict

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


class imgs():
    def __init__(self, im, gt, pr, dice, iou, hd):
        im=im.squeeze()
        im=im.cpu().detach().numpy()
        if len(im.shape)==2:
          im=np.stack((im,im,im),axis=0) 
        self.im=np.transpose(im, (1,2,0))
        
        gt=gt.squeeze()
        gt=gt.cpu().detach().numpy()
        self.gt=np.transpose(gt, (1,2,0))
        
        pr=pr.squeeze()
        pr=pr.cpu().detach().numpy()
        self.pr=np.transpose(pr, (1,2,0))
        
        self.dice=dice
        self.iou=iou
        self.hd=hd



def test(test_loader, model,
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
            
            ##Loss function
            # loss=loss_fn(preds, y)
            
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
                stream.set_description("Testing \t\t\t{metric_monitor}".format(metric_monitor=metric_monitor))

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
    
    # if verbose:        
    #     print(f"Got Dice score mean:\t {np.mean(dices):.8f}")
    #     print(f"Got Dice score max:\t {np.max(dices):.8f}")
    #     print(f"Got Dice score min:\t {np.min(dices):.8f}")
    #     print(f"Got Dice score std:\t {np.std(dices):.8f}")
    
    model.train()
    model=model.cpu()
    
    return dices, ious, hds, hds95 #, nsds

def overlay_imgs(inputs, masks, preds, alpha=0.4):
    #Lists to concatenate and recover a batch
    lst_masks=[]
    lst_preds=[]
    
    colors_mask = {"green":(0,255,0),
                   "blue":(0,0,255)} 
    colors_pred = {"yellow":(255,255,0),
                   "cyan":(0,255,255),} 
    
    
    if inputs.shape[1]==1:
        inputs=torch.cat((inputs,inputs,inputs), dim=1)
                
    inputs=inputs.to("cpu")
    preds=preds.to("cpu")
    
    # print("Input",inputs.shape, torch.unique(inputs))
    # print("Mask", masks.shape, torch.unique(masks))
    # print("Pred", preds.shape, torch.unique(preds))
    
    #The input
    for b in range(inputs.shape[0]):
        img=inputs[b,:,:,:]*255
        img=img.type(torch.uint8)
        # print("InputBatch",img.shape, torch.unique(img))
        
        img_inp = img
        img_pred = img
        #Mask and prediction on each channels
        for c, _kcm, _kcp in zip(range(masks.shape[1]), colors_mask.keys(), colors_pred.keys()):
            
            # img=inputs[b,c,:,:]*255
            # img=img.type(torch.uint8)
            # _color_msk=
            
            # print(c, _kcm, _kcp)
            
            mask = masks[b,c,:,:]#$.type(torch.bool)
            pred = preds[b,c,:,:]
            
            # print("InputChannel",img.shape, torch.unique(img), img.dtype)
            # print("PredChannel",pred.shape, torch.unique(pred), pred.dtype)
            # print("MaskChannel",mask.shape, torch.unique(mask), mask.dtype)
            
            # img_and_mask=draw_segmentation_masks(image=img_inp, masks=mask, 
            #                                       alpha=alpha, colors=(0,255,0))
            # img_and_pred=draw_segmentation_masks(image=img_pred, masks=pred, 
            #                                       alpha=alpha, colors=(255,0,255))
            
            img_and_mask=draw_segmentation_masks(image=img_inp, masks=mask, 
                                                  alpha=alpha, colors=colors_mask[_kcm])
            img_and_pred=draw_segmentation_masks(image=img_pred, masks=pred, 
                                                  alpha=alpha, colors=colors_pred[_kcp])
            
            # print("OverMask-Pred", img_and_mask.shape, img_and_pred.shape)
            
            img_inp = copy.deepcopy(img_and_mask)
            img_pred = copy.deepcopy(img_and_pred)
            
        lst_masks.append(img_and_mask.unsqueeze(dim=0))
        lst_preds.append(img_and_pred.unsqueeze(dim=0))
            
            # print("OverMask-Pred2", img_and_mask.shape, img_and_pred.shape)
        # img=inputs[b]*255
        # img=img.type(torch.uint8)
        
        # mask=masks[b].type(torch.bool)
        # pred=preds[b]
        
        # img_and_mask=draw_segmentation_masks(image=img, masks=mask, 
        #                                       alpha=alpha, colors=(0,255,0))
        # img_and_pred=draw_segmentation_masks(image=img, masks=pred, 
        #                                       alpha=alpha, colors=(255,0,255))
        # #print('Test',img.shape, img.dtype, pred.shape, pred.dtype, img_and_pred.shape, img_and_pred.dtype, torch.max(pred), torch.min(pred))
        # lst_masks.append(img_and_mask.unsqueeze(dim=0))
        # lst_preds.append(img_and_pred.unsqueeze(dim=0))
    
    #Recover a tensor BxCxHxW
    tensor_masks=torch.cat(lst_masks, dim=0)
    tensor_preds=torch.cat(lst_preds, dim=0)
    # print("OverMask-PredOut", tensor_masks.shape, tensor_preds.shape)
    # lst_masks.append(img_and_mask.unsqueeze(dim=0))
    # lst_preds.append(img_and_pred.unsqueeze(dim=0))
    # print("OverMask-Pred2", img_and_mask.shape, img_and_pred.shape)
    
    return tensor_masks, tensor_preds

def set_title(tensor,string):
    tensor=tensor.squeeze()
    numpy=tensor.cpu().detach().numpy()
    numpy=np.transpose(numpy, (1,2,0))
    pil_image=Image.fromarray(numpy)    
    
    
    font=ImageFont.truetype('Arial.ttf',12)
    # font=ImageFont.truetype(r'/home/202201016n/serverBUAP/NASGP-Net/Arial.ttf', 12)
    # font=ImageFont.truetype(r'/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 25)
    
    w, h = font.getsize(string)
    draw=ImageDraw.Draw(pil_image)
    wim, him = pil_image.size
    draw.text(((wim-w)/2, 15*(him-h)/16), string, font=font, fill='blue')
    numpy=np.array(pil_image)
    numpy=np.transpose(numpy, (2,0,1))
    tensor=torch.from_numpy(numpy)
    tensor=tensor.unsqueeze(dim=0) 
    
    return tensor

def set_title2(tensor,string):
    # font=ImageFont.truetype(r'/home/202201016n/serverBUAP/NASGP-Net/Arial.ttf', 12)
    font=ImageFont.truetype('Arial.ttf',12)
    w, h = font.getsize(string)
    out_tensor=[]
    for i in range(tensor.shape[0]):
        t=tensor[i]
        numpy=t.cpu().detach().numpy()
        numpy=np.transpose(numpy, (1,2,0))
        pil_image=Image.fromarray(numpy)    
    
        draw=ImageDraw.Draw(pil_image)
        wim, him = pil_image.size
        draw.text(((wim-w)/2, 15*(him-h)/16), string, font=font, fill='blue')
        numpy=np.array(pil_image)
        
        numpy=np.transpose(numpy, (2,0,1))
        t=torch.from_numpy(numpy)
        t=tensor.unsqueeze(dim=0)
        out_tensor.append(t)
    # out_tensor=torch.cat(out_tensor, dim=0)
    # print(out_tensor.shape)
    return out_tensor

def save_grid(objs, num, option, ruta):
    if option == 'best':
        lstBest=sorted(objs, key = lambda x: x.dice, reverse=True)
        bests=lstBest[:num]
    if option == 'worst':
        lstWorst=sorted(objs, key = lambda x: x.dice)
        bests=lstWorst[:num]
        bests.reverse()
    figure, ax = plt.subplots(nrows=num, ncols=3, figsize=(10,10))
    
    for i, best in enumerate(bests):
        ax[i,0].imshow(best.im)
        ax[i,1].imshow(best.gt)
        ax[i,2].imshow(best.pr)
        
        ax[i,0].set_title("Image " + option + " " +str(i+1))
        ax[i,1].set_title("Ground " + option + " " + str(i+1))
        ax[i,2].set_title("Predicted " + option + " " +str(i+1))
        
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    
    plt.savefig(ruta+"/Three Segmentations_"+option+".png", bbox_inches='tight')
    plt.close(figure)
    plt.show()
