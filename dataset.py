import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

def get_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
class PolypDataset(Dataset):
    
    def __init__(self,image_paths,mask_paths,transform=None):
        self.image_paths=image_paths
        self.mask_paths=mask_paths
        self.transform=transform       
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self,idx):
        image=cv2.imread(self.image_paths[idx])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.mask_paths[idx],cv2.IMREAD_GRAYSCALE)
        mask=(mask>127).astype(np.float32)
        if self.transform:
            augmented=self.transform(image=image,mask=mask)
            image=augmented['image']
            mask=augmented['mask'].unsqueeze(0)
        return image,mask
def get_dataloaders():
    image_files=sorted(os.listdir(config.IMAGE_DIR ))
    mask_files=sorted(os.listdir(config.MASK_DIR))
    image_path= [os.path.join(config.IMAGE_DIR,f) for f in image_files]
    mask_path= [os.path.join(config.MASK_DIR,f) for f in mask_files]
    train_imgs, val_imgs,train_masks, val_masks = train_test_split(
        image_path , mask_path,
        test_size=config.VAL_SPLIT,
        random_state=42
    )
    train_dataset=PolypDataset(train_imgs, train_masks, get_transforms(is_train=True))
    val_dataset=PolypDataset(val_imgs, val_masks, get_transforms(is_train=False))
    train_loader= DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader