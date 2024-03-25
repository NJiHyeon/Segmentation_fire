import torch
import numpy as np
import pandas as pd
from model import UNet
import rasterio
import joblib
from torchvision import transforms
from tqdm import tqdm
# ==================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check
# ===================================================================================
is_cuda = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
MAX_PIXEL_VALUE = 65535
NUM_CLASSES = 1
IMAGE_SIZE = 256
ckpt_path = "./train_output/unet_baseline/model_unet_32.pth"
test_meta = pd.read_csv('../data/test_meta.csv')
# ===================================================================================
def load_model(ckpt_path, num_classes, device) :
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = UNet(num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
model = load_model(ckpt_path, NUM_CLASSES, DEVICE)


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@torch.no_grad()
def predict_segment(test_meta) :
    y_pred_dict = {}
    n = 0
    for i in tqdm(test_meta['test_img']):
        m = 0
        img = rasterio.open(f'../data/test_img/{i}').read((7,6,2)).transpose((1, 2, 0))
        img = np.float32(img)/MAX_PIXEL_VALUE #[256, 256, 3]
        img = transformer(img) #[3, 256, 256]
        img = img.to(DEVICE) 
        
        if NUM_CLASSES > 1:
            y_pred = model(img.unsqueeze(dim=0)) 
            y_pred = torch.argmax(y_pred.squeeze(dim=0).cpu(), dim=0) #[256, 256], argmax말고 임계값을 줄 수도 있음
            y_pred = y_pred.numpy()
            y_pred = y_pred.astype(np.uint8)
        else :
            y_pred = model(img.unsqueeze(dim=0)) #[1,1,256,256]
            y_pred = y_pred.data.cpu().numpy() #sigmoid
            #y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
            y_pred = np.where(y_pred[0, 0, :, :] > 0.15, 1, 0)
            y_pred = y_pred.astype(np.uint8)
            y_pred_dict[i] = y_pred
            n += np.sum(y_pred)
    print('n', n)
    joblib.dump(y_pred_dict, './unet_32_015.pkl')

predict_segment(test_meta=test_meta)



