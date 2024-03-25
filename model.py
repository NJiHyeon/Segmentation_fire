import torch.nn as nn
import torch

def ConvLayer(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return layers


class Encoder(nn.Module) :
    def __init__(self, ) :
        super().__init__()
        self.conv_block1 = ConvLayer(in_channels=3, out_channels=64)
        self.conv_block2 = ConvLayer(in_channels=64, out_channels=128)
        self.conv_block3 = ConvLayer(in_channels=128, out_channels=256)
        self.conv_block4 = ConvLayer(in_channels=256, out_channels=512)
        self.conv_block5 = ConvLayer(in_channels=512, out_channels=1024)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x) :
        encode_features = []
        out = self.conv_block1(x)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block2(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block3(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block4(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block5(out)
        return out, encode_features
    
    
def UpConvLayer(in_channels, out_channels):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return layers

class Decoder(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.upconv_layer1 = UpConvLayer(in_channels=1024, out_channels=512)
        self.conv_block1 = ConvLayer(in_channels=512+512, out_channels=512)
        
        self.upconv_layer2 = UpConvLayer(in_channels=512, out_channels=256)
        self.conv_block2 = ConvLayer(in_channels=256+256, out_channels=256)
        
        self.upconv_layer3 = UpConvLayer(in_channels=256, out_channels=128)
        self.conv_block3 = ConvLayer(in_channels=128+128, out_channels=128)
        
        self.upconv_layer4 = UpConvLayer(in_channels=128, out_channels=64)
        self.conv_block4 = ConvLayer(in_channels=64+64, out_channels=64)
        
    def forward(self, x, encoder_features) :
        out = self.upconv_layer1(x)
        out = torch.cat([out, encoder_features[-1]], dim=1)
        out = self.conv_block1(out)
        
        out = self.upconv_layer2(out)
        out = torch.cat([out, encoder_features[-2]], dim=1)
        out = self.conv_block2(out)
        
        out = self.upconv_layer3(out)
        out = torch.cat([out, encoder_features[-3]], dim=1) #[B, F, H, W]
        out = self.conv_block3(out)
        
        out = self.upconv_layer4(out)
        out = torch.cat([out, encoder_features[-4]], dim=1) #[B, F, H, W]
        out = self.conv_block4(out)
        
        return out


import torch.nn.functional as F

class UNet(nn.Module) :
    def __init__(self, num_classes) :
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x) :
        out, encoder_features = self.encoder(x)
        out = self.decoder(out, encoder_features)
        out = self.head(out)
        #out = self.sigmoid(out)
        return out