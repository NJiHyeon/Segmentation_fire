import os
import torch
import cv2
import rasterio
import tqdm
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.utils import shuffle as shuffle_lists
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

from model import UNet
from utils import dice_loss, focal_loss
from utils import Cutout

from dataloader import build_dataloader
#================================================================================================================
# 하이퍼파라미터
NUM_CLASSES = 1
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 256
RANDOM_STATE = 42 
MAX_PIXEL_VALUE = 65535

# 사용할 데이터의 meta정보 가져오기
data_dir = "./data/"
train_meta = pd.read_csv('./data/train_meta.csv')
test_meta = pd.read_csv('./data/test_meta.csv')

save_name = 'base_line'
IMAGES_PATH = './data/train_img/'
MASKS_PATH = './data/train_mask/'
OUTPUT_DIR = './train_output/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# GPU 설정
CUDA_DEVICE = 0
torch.cuda.set_device(CUDA_DEVICE)
torch.manual_seed(torch.initial_seed())
is_cuda = True
DEVICE = torch.device('cuda' if torch.cuda.is_available and is_cuda else 'cpu')
#================================================================================================================
##Train##
def train_one_epoch(dataloaders, model, criterion, optimizer, device) :
    losses = {}

    for phase in ["train", "val"]:
        running_loss = 0.0

        if phase == "train":
            model.train()
        else:
            model.eval()

        for index, batch in enumerate(dataloaders[phase]):
            images = batch[0].to(device) # [12, 3, 256, 256]
            targets = batch[1].to(device) # [12, 256, 256]

            with torch.set_grad_enabled(phase == "train"):
                # num_class에 따라 squeeze할지, 말지 ; 일단1이니까 squeeze먼저 진행
                predictions = model(images)
                
                loss_bce = criterion(predictions.squeeze(1), targets.squeeze(1))
                loss_dice = dice_loss(F.sigmoid(predictions.squeeze(1)), targets.squeeze(1), multiclass=False)
                loss_focal = focal_loss(predictions.squeeze(1), targets.squeeze(1))
                loss = loss_bce + loss_dice + loss_focal
                
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()

            if phase == "train":
                
                if index % 100 == 0:
                    text = f"{index}/{len(dataloaders[phase])}" + \
                            f" - Running Loss: {loss.item():.4f}"
                    print(text)
     
        losses[phase] = running_loss / len(dataloaders[phase])
    return losses
#============================================================================================================
##Train Run!!##
dataloaders = build_dataloader(train_meta, batch_size=BATCH_SIZE)
resume_from =  None
if resume_from :
    model_data = torch.load(resume_from)
    model = UNet(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(model_data['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_data['optimizer_state_dict'])
    start_epoch = model_data['epoch'] + 1
else :
    model = UNet(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    start_epoch = 1


criterion = nn.CrossEntropyLoss() if NUM_CLASSES > 1 else nn.BCEWithLogitsLoss()

train_loss = []
val_loss = []

for epoch in range(EPOCHS) :
    losses = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
    train_loss.append(losses["train"])
    val_loss.append(losses["val"])
    
    print(f"{epoch}/{EPOCHS} - Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

    save_dir = "./train_output/unet_transform2"
    model_name = f"model_unet_{epoch:02d}.pth"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss},
               os.path.join(save_dir, model_name))