# DataLoader baseline : transform X

import os
import torch
import rasterio
import numpy as np
import pandas as pd
from torchvision import transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

data_dir = "./data/"
train_meta = pd.read_csv('./data/train_meta.csv')
test_meta = pd.read_csv('./data/test_meta.csv')
IMAGES_PATH = './data/train_img/'
MASKS_PATH = './data/train_mask/'
MAX_PIXEL_VALUE = 65535
RANDOM_STATE = 42

# Dataset
class SPARK_dataset() :
    def __init__(self, x_tr, x_val, phase, transformer=None) :
        self.phase = phase
        if phase == "train" :
            self.image_files = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
            self.mask_files = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]
            #self.transformer = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #Cutout(n_holes=, length=)])
        else :
            self.image_files = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
            self.mask_files = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]
            #self.transformer = transforms.Compose([
                #transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transformer = transformer
    
    def fopen_image(self, path) :
        img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0)) #[3, 256, 256] -> [256, 256, 3]
        img = np.float32(img) / MAX_PIXEL_VALUE
        return img
        
    def fopen_mask(self, path) :
        img = rasterio.open(path).read().transpose((1, 2, 0)) #[1, 256, 256] -> [256, 256, 1]
        seg = np.float32(img)
        return seg
    
    def __len__(self, ) :
        return len(self.image_files)
    
    def __getitem__(self, index) :
        # image
        image = self.fopen_image(self.image_files[index]) 
        if self.transformer :
            image = self.transformer(image) #[3, 256, 256]
        
        # mask
        mask = self.fopen_mask(self.mask_files[index]) # [256, 256, 1]
        mask = mask.squeeze(-1) # [256, 256]
        target = torch.from_numpy(mask) #.long()
        return image, target
    
    
def build_transformer() :
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer


def collate_fn(batch) :
    images = []
    targets = []
    for a, b in batch :
        images.append(a)
        targets.append(b)
        
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


def build_dataloader(train_meta, batch_size=4) :
    x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
    transformer = build_transformer()
    
    dataloaders = {}
    
    train_dataset = SPARK_dataset(x_tr, x_val, phase="train", transformer=transformer)
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    val_dataset = SPARK_dataset(x_tr, x_val, phase="val", transformer=transformer)
    dataloaders["val"] = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    return dataloaders