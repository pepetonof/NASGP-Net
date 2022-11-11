from torch.utils.data import Dataset
from skimage import io
#For multiclass segmentation and CrossEntropyLoss
class DataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
        
        image=io.imread(img_id)
        mask =io.imread(mask_id)
        mask[mask==255]=1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        return image, mask
