'''
직관적인 모델 구조 구현
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def convBlockx2(in_channels, out_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return layer

class Encoder(nn.Module): # 인코더 구조
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encode = convBlockx2(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down_conv(x) 
        x = self.pool(x)
        
        return x

class Decoder(nn.Module): # 디코더 구조
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.decode(x)

        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)
        
        self.mid = Decoder(512, 1024, 512)

        self.dec4 = Decoder(1024, 512, 256)
        self.dec3 = Decoder(512, 256, 128)
        self.dec2 = Decoder(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        mid = self.mid(enc4)
        dec4 = self.dec4(torch.concat([mid, enc4], dim=1)) # skip connection
        dec3 = self.dec3(torch.concat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.concat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.concat([dec2, enc1], dim=1))
        final = self.final(dec1)

        return final
