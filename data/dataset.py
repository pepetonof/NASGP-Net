from torch.utils.data import Dataset
from skimage import io
import numpy as np

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

    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        
        image=io.imread(img_id)
        mask =io.imread(mask_id)
        # print(mask.dtype)
        
        
        no_classes = len(np.unique(mask))
        interval = 256/(no_classes-1)
        self.vals_class = np.array([round(i * interval) for i in range(no_classes)])
        self.vals_class[1:] -= 1
        
        # print('Vals_class:\t',  self.vals_class)
        # print('Vals_mask:\t', self.vals_mask)
        
        # print('DifferentValuesRead:\t', np.unique(mask))
        # print(image.shape, mask.shape)
        
        
        # for val_p, val_msk in zip(self.vals_class[1:], self.vals_mask[1:]):
        #     mask[mask==val_p] = 1.0 #val_msk
        #     print('val_changed', val_p)
        
        for val_p in self.vals_class[1:]:
            mask[mask==val_p] = 1.0
            # print('val_changed', val_p)
            
        
        # print('DifferentValuesReadChanged:\t', np.unique(mask))
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        return image, mask
