from torch.utils.data import Dataset
from skimage import io
import numpy as np

import nrrd
import nibabel as nib
import torch

#For multiclass segmentation and CrossEntropyLoss
class DataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, no_classes=2):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        #Adapt to the number of clases/out_channels
        # self.no_clases = n
        # interval = 256/(no_classes-1)
        # self.vals_class = np.array([round(i * interval) for i in range(no_classes)])
        # self.vals_class[1:] -= 1
        
        self.vals_mask=np.array(list(range(no_classes)))
        
        # no_classes = len(np.unique(mask))
        interval = 256/(no_classes-1)
        self.vals_class = np.array([round(i * interval) for i in range(no_classes)])
        self.vals_class[1:] -= 1

    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        
        image=io.imread(img_id)
        mask =io.imread(mask_id)
        # print(image.shape, mask.shape)
        
        # no_classes = len(np.unique(mask))
        # interval = 256/(no_classes-1)
        # self.vals_class = np.array([round(i * interval) for i in range(no_classes)])
        # self.vals_class[1:] -= 1
        
        # print('Vals_class:\t',  self.vals_class)
        # print('Vals_mask:\t', self.vals_mask)
        
        #Multiclass
        # for val_p, val_msk in zip(self.vals_class[1:], self.vals_mask[1:]):
        #     mask[mask==val_p] = 1.0 #val_msk
        #     print('val_changed', val_p)
        
        #Join ROI
        for val_p in self.vals_class[1:]:
            mask[mask==val_p] = 1.0
            # print('val_changed', val_p)
        
        # print('DifferentValuesReadChanged:\t', np.unique(mask))
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        return image, mask

class Dataset3D(Dataset):
    def __init__(self, image_dir:list, mask_dir:list, transform=None, no_classes=2):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.vals_mask=np.array(list(range(no_classes)))
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        vol_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        if str(vol_id)[-4:]=='.png':
            raise TypeError('Data is not 3D')
            # vol=io.imread(vol_id)
            # mask =io.imread(mask_id)
        elif str(vol_id)[-3:]=='.gz':
            vol =  nib.load(vol_id).get_fdata().astype(np.uint8)
            mask = nib.load(mask_id).get_fdata().astype(np.uint8)
        elif str(vol_id)[-5:]=='.nrrd':
            vol = nrrd.read(vol_id)
            mask = nrrd.read(mask_id)
            
        # print(vol.shape, vol.max(), vol.min(), vol.dtype)
        # print(mask.shape, mask.max(), mask.min(), mask.dtype, np.unique(mask)) 
        
        #Select one dimension for 4D
        if (len(vol.shape)>3):
            vol = vol[:,:,:,1]
        
        #Join ROI
        for val_p in self.vals_mask[1:]:
            mask[mask==val_p] = 1.0
        
        #Apply transform
        if self.transform is not None:
            augmentations = self.transform(image=vol, mask=mask)
            # print(vol.shape, vol.max(), vol.min(), vol.dtype)
            # print(mask.shape, mask.max(), mask.min(), mask.dtype)
            vol=augmentations["image"]
            mask=augmentations["mask"]
            # print('Transform')
            # print(vol.shape, vol.max(), vol.min(), vol.dtype)
            # print(mask.shape, mask.max(), mask.min(), mask.dtype, np.unique(mask))
        
        return vol, mask
    
class Dataset3D22D(Dataset):
    def __init__(self, image_data, mask_data):
        # print(image_data.shape, image_data.max(), image_data.min(), image_data.dtype)
        # print(mask_data.shape, mask_data.max(), mask_data.min(), mask_data.dtype, np.unique(mask_data)) 
        # print(image_data.shape, mask_data.shape)
        
        #Select slices where the masks with non zero slices
        non_zero_slc = (torch.amax(mask_data, dim=(1,2))==1).nonzero(as_tuple=False).squeeze()
        # print('nonzero', non_zero_slc.shape)
        
        self.image_data = torch.index_select(image_data, 0, non_zero_slc)
        self.mask_data = torch.index_select(mask_data, 0, non_zero_slc)
        #Tensor Format NXCXHXW
        self.image_data= torch.unsqueeze(self.image_data, 1)
        # self.mask_data = torch.unsqueeze(self.mask_data, 1)
        
        # print('ImageDataShape',self.image_data.shape, image_data.shape)
        # print('MaskDataShape',self.mask_data.shape, mask_data.shape)
        
        # print(self.image_data.shape, self.image_data.max(), self.image_data.min(), self.image_data.dtype)
        # print(self.mask_data.shape, self.mask_data.max(), self.mask_data.min(), self.mask_data.dtype, np.unique(self.mask_data)) 
    
    def __len__(self):
        return self.image_data.shape[0]
    
    def __getitem__(self, index:int):
        return self.image_data[index, :, :, :], self.mask_data[index, :, :]
    
class Dataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        # print('IMG_ID', img_id)
        # print('MASK_ID', mask_id)
        
        image=io.imread(img_id)
        mask =io.imread(mask_id)
        
        # print('IMAGE', image.shape, image.max(), image.min(), image.dtype, type(image), np.unique(image))
        # print('MASK', mask.shape, mask.max(), mask.min(), mask.dtype, type(mask), np.unique(mask)) 
        
        mask[mask==255]=1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        # print('IMAGET', img_id, index, image.shape, image.max(), image.min(), image.dtype)
        # print('MASKT', index, mask.shape, mask.max(), mask.min(), mask.dtype, np.unique(mask)) 
        
        return image, mask