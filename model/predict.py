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
import os
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms.functional as TF
# import torchvision.transforms.functionalJ as TF
import matplotlib.pyplot as plt
import copy
from torchvision.utils import make_grid
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
    def __init__(self, im, gt, pr, dice):#, iou, hd):
        self.im = im
        self.gt = gt
        self.pr = pr
        
        self.dice=dice
        # self.iou=iou
        # self.hd=hd


def test(test_loader, model, metrics,
         save_imgs=False, ruta="saved_images/", 
         device="cuda", verbose=False, fold=None):
    
    model = model.to(device)
    
    """Verbose"""
    metric_monitor=MetricMonitor()
    model.eval()
    if verbose:
        stream = tqdm(test_loader)
    else:
        stream = test_loader
    
    """Metrics to compare"""
    metricsTest = {}
    for m in metrics:
        key=str(type(m)).strip('>').strip("'").split('.')[-1]
        metricsTest[key]=[]
        if key == 'HDMetric':
            metricsTest[key+'95']=[]
    # nsds = []
    
    """For save  three best predictions"""
    objs=[]
    
    
    with torch.no_grad():
    
        for idx, (x, y) in enumerate(stream, start=1):
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
        
            #Takes each region of interest
            pred_soft = F.softmax(preds, dim=1)
            
            if pred_soft.shape[1:]!=y.shape[1:]:
                pred_soft = TF.resize(pred_soft, size=y.shape[1:])
            
            """Metrics for evaluation"""
            for m in metrics:
                key=str(type(m)).strip('>').strip("'").split('.')[-1]
                score = m(pred_soft, y)
                if key == 'HDMetric':  
                    metricsTest[key].append(score[0])
                    metricsTest[key+'95'].append(score[1])
                else:
                    metricsTest[key].append(score)
            
            """Metrics for evaluation"""            
            if save_imgs:
                filename_img = test_loader.dataset.image_dir[idx-1].name.split('.')[0]
                # filename_mask = test_loader.dataset.mask_dir[idx-1].name.split('.')[0]
                # print()
                # print('IMGS',filename_img, type(filename_img))
                # print('MASKS',filename_mask, type(filename_img))
                # path_imgs=ruta+"/test_segmentation"
                path_imgs = f"{ruta}/test_segmentation/fold_{fold}" if fold!=None else f"{ruta}/test_segmentation"
                if not os.path.exists(path_imgs):
                    os.makedirs(path_imgs)
                
                sem_classes=['__background__'] + ["roi" + str(i) for i in range(1,preds.shape[1])]
                sem_class_to_idx={cls:idx for (idx, cls) in enumerate(sem_classes)} 

                #For each class
                y_roi = []
                pred_roi = []
                for key in list(sem_class_to_idx.keys())[1:]:
                    # class_dim = sem_class_to_idx[key]
                    pbool = pred_soft.argmax(1) == sem_class_to_idx[key]
                    ybool = y ==sem_class_to_idx[key]
                    
                    pred_roi.append(pbool.unsqueeze(dim=1))
                    y_roi.append(ybool.unsqueeze(dim=1))
                    
                pred_roi = torch.cat(pred_roi, dim=1)
                y_roi = torch.cat(y_roi, dim=1)
                # print("Pred_ROI", pred_roi.shape, torch.unique(pred_roi))
                # print("Y_ROI", y_roi.shape, torch.unique(y_roi))
                
                #Obtain original images with masks and predictions on top
                over_masks, over_preds = overlay_imgs(x, y_roi, pred_roi)
                
                #Save original image
                torchvision.utils.save_image(x, f"{path_imgs}/{filename_img}.png")
                over_masks = (over_masks.float())/255.00
                title = f"{path_imgs}/{filename_img}_mask_F{fold}.png" if fold!=None else f"{path_imgs}/{filename_img}_mask.png"
                torchvision.utils.save_image(over_masks, title)
                
                #Set dice index to the over_preds
                dice = metricsTest['DiceMetric'][-1]
                over_preds=set_title(over_preds, 'Dice='+str(round(dice,3)))
                
                #For save with torch.utils.save_image()
                over_preds = (over_preds.float())/255.00
                title = f"{path_imgs}/{filename_img}_pred_F{fold}_dice={round(dice,3)}.png" if fold else f"{path_imgs}/{filename_img}_pred_dice={round(dice,3)}.png"
                torchvision.utils.save_image(over_preds, title)
            
                #Save grid of the three best/worst individuals
                # iou = metricsTest['IoUMetric'][-1]
                # hd = metricsTest['HDMetric95'][-1]
                obj_best=imgs(x, over_masks, over_preds, dice)#, iou, hd)
                objs.append(obj_best)
            
            if verbose:
                """Monitor Dice for Test Set Validation"""
                dice = metricsTest['DiceMetric'][-1]
                metric_monitor.update("dice", dice)#loss.item())
                stream.set_description("Testing \t\t\t{metric_monitor}".format(metric_monitor=metric_monitor))

            #For saving memory
            del x, y, preds
            torch.cuda.empty_cache()
    
    #Save grid of the three best/worst individual
    if save_imgs:
        save_grid(objs, 3, 'best', f"{path_imgs}/best_seg.png")
        save_grid(objs, 3, 'worst', f"{path_imgs}/worst_seg.png")
    
    model.train()
    model=model.cpu()
    
    return metricsTest

def overlay_imgs(inputs, masks, preds, alpha=0.4):
    #Lists to concatenate and recover a batch
    lst_masks=[]
    lst_preds=[]
    
    colors_mask = {"green":(0,255,0),
                   "blue":(0,0,255)} 
    colors_pred = {"magenta":(255,0,255),
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
            
            img_and_mask=draw_segmentation_masks(image=img_inp, masks=mask, 
                                                  alpha=alpha, colors=colors_mask[_kcm])
            img_and_pred=draw_segmentation_masks(image=img_pred, masks=pred, 
                                                  alpha=alpha, colors=colors_pred[_kcp])
            
            # print("OverMask-Pred", img_and_mask.shape, img_and_pred.shape)
            
            img_inp = copy.deepcopy(img_and_mask)
            img_pred = copy.deepcopy(img_and_pred)
            
        lst_masks.append(img_and_mask.unsqueeze(dim=0))
        lst_preds.append(img_and_pred.unsqueeze(dim=0))
            
    
    #Recover a tensor BxCxHxW
    tensor_masks=torch.cat(lst_masks, dim=0)
    tensor_preds=torch.cat(lst_preds, dim=0)
    # print("OverMask-PredOut", tensor_masks.shape, tensor_preds.shape)
    # lst_masks.append(img_and_mask.unsqueeze(dim=0))
    # lst_preds.append(img_and_pred.unsqueeze(dim=0))
    # print("OverMask-Pred2", img_and_mask.shape, img_and_pred.shape)
    
    return tensor_masks, tensor_preds

def set_title(tensor,string):
    # font=ImageFont.truetype('C:/Users/josef/OneDrive - Universidad Veracruzana/DIA/NASGP-Net/code/Arial.ttf',11)
    # font=ImageFont.truetype(r'/home/202201016n/serverBUAP/NASGP-Net/Arial.ttf', 12)
    font=ImageFont.truetype('Arial.ttf',11)
    w, h = font.getsize(string)
    out_tensor=[]
    # print(tensor.shape)
    for i in range(tensor.shape[0]):
        t=tensor[i]
        
        # print(t.shape)
        numpy=t.cpu().detach().numpy()
        numpy=np.transpose(numpy, (1,2,0))
        pil_image=Image.fromarray(numpy)    
    
        draw=ImageDraw.Draw(pil_image)
        wim, him = pil_image.size
        draw.text(((wim-w)/2, 15*(him-h)/16), string, font=font, fill='blue')
        numpy=np.array(pil_image)
        # print(numpy.shape)
        
        numpy=np.transpose(numpy, (2,0,1))
        t=torch.from_numpy(numpy)
        # print(t.shape)
        t=torch.unsqueeze(t, dim=0)#tensor.unsqueeze(dim=)
        # print(t.shape)
        out_tensor.append(t)
    out_tensor=torch.cat(out_tensor, dim=0)
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
        
    figure, ax = plt.subplots(nrows=num, ncols=3)
    for i, obj in enumerate(bests):
        grid_im=make_grid(obj.im)
        grid_gt=make_grid(obj.gt)
        grid_pr=make_grid(obj.pr)
        # print(grid_im.shape, grid_gt.shape, grid_pr.shape)
        
        ax[i,0].imshow(np.transpose(grid_im.cpu().detach().numpy(), (1,2,0)))
        ax[i,1].imshow(np.transpose(grid_gt.cpu().detach().numpy(), (1,2,0)))
        ax[i,2].imshow(np.transpose(grid_pr.cpu().detach().numpy(), (1,2,0)))
        
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    
    plt.savefig(ruta, bbox_inches='tight', dpi=600)
    plt.close(figure)
    plt.show()
    
