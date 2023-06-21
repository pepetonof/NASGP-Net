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
import matplotlib.pyplot as plt

from model.loss_f import hdistance_loss, IoU_loss, dice_loss
from model.train_valid import MetricMonitor

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
    

def test(test_loader, model, loss_fn, 
         save_imgs=False, ruta="saved_images/", device="cuda", verbose=False):
    
    metric_monitor=MetricMonitor()
    model.eval()
    if verbose:
        stream = tqdm(test_loader)
    else:
        stream = test_loader
    num_correct=0
    num_pixels=0
    
    """Metrics to compare"""
    dices, ious, hds = [],[],[]
    """For save  three best predictions"""
    objs=[]
    
    # if save_imgs:
    #     print('=> Saving images...')
    
    with torch.no_grad():
    
        for idx, (x, y) in enumerate(stream, start=1):
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
        
            #Takes the placenta class
            pbool = F.softmax(preds, dim=1)
            sem_classes=['__background__', 'placenta']
            sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)}
            class_dim=1
            pbool = pbool.argmax(class_dim) == sem_class_to_idx['placenta']
            
            if pbool.shape[1:]!=y.shape[1:]:
                pbool = TF.resize(pbool, size=y.shape[1:])
            
            """Metrics for evaluation"""
            #Accuracy
            num_correct += (pbool==y.unsqueeze(1)).sum()
            num_pixels += torch.numel(pbool)
            
            if preds.shape[2:]!=y.shape[1:]:
                preds = TF.resize(preds, size=y.shape[1:])
            
            #Dice coefficient
            dice = dice_loss(preds, y)
            dice = 1-dice
            dice = dice.detach().item()
            dices.append(dice) #Append to dices
            
            ##Loss function
            loss=loss_fn(preds, y)
            
            # """Monitor Loss for Test Set Validation"""
            # metric_monitor2.update("Loss:", loss)
            # stream.set_description("Testing. {metric_monitor}".format(metric_monitor=metric_monitor2))
        
            #IoU
            iou = IoU_loss(preds, y)
            iou = 1-iou
            iou =iou.detach().item()
            ious.append(iou)#Append to ious
            
            #Hausdorff Distance
            hd = hdistance_loss(preds, y)
            hds.append(hd)
            
            if save_imgs:
                #Save each image
                #Save original image
                torchvision.utils.save_image(x, f"{ruta}/{idx}_in.png")
                #Obtain original images with masks and predictions on top
                over_masks, over_preds = overlay_imgs(x, y, pbool)
                #Set dice index to the over_preds
                over_preds=set_title(over_preds, 'Dice='+str(round(dice,6)))
                
                #For save with torch.utils.save_image()
                over_masks = (over_masks.float())/255.00
                over_preds = (over_preds.float())/255.00
                torchvision.utils.save_image(over_masks, f"{ruta}/{idx}_overmask.png")
                torchvision.utils.save_image(over_preds, f"{ruta}/{idx}_overprd.png")
            
                #Save grid of the three best/worst individuals
                obj_best=imgs(x, over_masks, over_preds, dice, iou, hd)
                objs.append(obj_best)
    
    
            #For saving memory
            del x, y, preds
            torch.cuda.empty_cache()
        
    if verbose:
        """Monitor Dice for Test Set Validation"""
        metric_monitor.update("Loss", loss.item())
        stream.set_description("Testing. \t\t{metric_monitor}".format(metric_monitor=metric_monitor))
    
    #Save grid of the three best/worst individual
    if save_imgs:
        save_grid(objs, 3, 'best', ruta)
        save_grid(objs, 3, 'worst', ruta)
    
    if verbose:        
        print(f"Got Dice score mean:\t {np.mean(dices):.8f}")
        print(f"Got Dice score max:\t {np.max(dices):.8f}")
        print(f"Got Dice score min:\t {np.min(dices):.8f}")
        print(f"Got Dice score std:\t {np.std(dices):.8f}")
        print(f"Got IoU score mean:\t {np.mean(ious):.8f}")
        print(f"Got IoU score max:\t {np.max(ious):.8f}")
        print(f"Got IoU score min:\t {np.min(ious):.8f}")
        print(f"Got IoU score std:\t {np.std(ious):.8f}")
        print(f"Got H. distance mean:\t {np.mean(hds):.8f}")
        print(f"Got H. distance max:\t {np.max(hds):.8f}")
        print(f"Got H. distance min:\t {np.min(hds):.8f}")
        print(f"Got H. distance std:\t {np.std(hds):.8f}")
    # print(f"Got Accuracy:\t\t {num_correct/num_pixels*100:.3f} %")
    
    model.train()
    
    return dices, ious, hds

def overlay_imgs(inputs, masks, preds, alpha=0.4):
    #Lists to concatenate and recover a batch
    lst_masks=[]
    lst_preds=[]
    
    if inputs.shape[1]==1:
        inputs=torch.cat((inputs,inputs,inputs), dim=1)
                
    inputs=inputs.to("cpu")
    preds=preds.to("cpu")
    #print(inputs.shape, inputs.dtype, preds.shape, preds.dtype)
    
    #Si se trata de escala de grises
    #if inputs.shape[1]!=3:
    #    inputs=torch.cat((inputs, inputs, inputs),dim=1)

    for i in range(inputs.shape[0]):
        img=inputs[i]*255
        img=img.type(torch.uint8)
        
        mask=masks[i].type(torch.bool)
        pred=preds[i]
        img_and_mask=draw_segmentation_masks(image=img, masks=mask, 
                                             alpha=alpha, colors=(0,255,0))
        img_and_pred=draw_segmentation_masks(image=img, masks=pred, 
                                             alpha=alpha, colors=(255,0,0))
        #print('Test',img.shape, img.dtype, pred.shape, pred.dtype, img_and_pred.shape, img_and_pred.dtype, torch.max(pred), torch.min(pred))
        lst_masks.append(img_and_mask.unsqueeze(dim=0))
        lst_preds.append(img_and_pred.unsqueeze(dim=0))
    
    #Recover a tensor BxCxHxW
    tensor_masks=torch.cat(lst_masks, dim=0)
    tensor_preds=torch.cat(lst_preds, dim=0)
    
    return tensor_masks, tensor_preds

def set_title(tensor,string):
    tensor=tensor.squeeze()
    numpy=tensor.cpu().detach().numpy()
    numpy=np.transpose(numpy, (1,2,0))
    pil_image=Image.fromarray(numpy)    
    
    
    # font=ImageFont.truetype('arial.ttf',25)
    font=ImageFont.truetype(r'/home/202201016n/serverBUAP/NASGP-Net/Arial.ttf', 12)
    #font=ImageFont.truetype(r'/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 25)
    
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
    font=ImageFont.truetype('arial.ttf',12)
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
    out_tensor=torch.cat(out_tensor, dim=0)
    
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
