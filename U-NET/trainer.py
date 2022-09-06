import os
from turtle import st
from unittest import skip
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import CityDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class UnetModel(pl.LightningModule):
    def __init__(self, params):
        super(UnetModel, self).__init__()
        self.n_classes = params.n_classes

        def convBlockx2(in_channels, out_channels):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

            return layer

        class Encoder(nn.Module): 
            def __init__(self, in_channels, out_channels):
                super(Encoder, self).__init__()
                self.encode = convBlockx2(in_channels, out_channels)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                x = self.encode(x) 
                skip = x
                x = self.pool(x)
                
                return x, skip

        class Decoder(nn.Module): 
            def __init__(self, in_channels, out_channels):
                super(Decoder, self).__init__()
                self.decode = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    
                )
                self.skip_connection = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x, skip):
                x = self.decode(x)
                x = self.skip_connection(torch.cat([x, skip], dim = 1))

                return x

        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)
        
        self.mid = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64)
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x : [-1, 3, 256, 256]
        enc1, skip1 = self.enc1(x) # [-1, 64, 128, 128]
        enc2, skip2 = self.enc2(enc1) # [-1, 128, 64, 64]
        enc3, skip3 = self.enc3(enc2) # [-1, 256, 32, 32]
        enc4, skip4 = self.enc4(enc3) # [-1, 512, 16, 16]
        
        mid = self.mid(enc4) # [-1, 1024, 16, 16]

        dec4 = self.dec4(mid, skip4) # [-1, 512, 32, 32]
        dec3 = self.dec3(dec4, skip3) # [-1, 256, 64, 64]
        dec2 = self.dec2(dec3, skip2) # [-1, 128, 128, 128]
        dec1 = self.dec1(dec2, skip1) # [-1, 64, 256, 256]
        
        final = self.final(dec1) # [-1, n_classes, 256, 256]

        return final

    def training_step(self, batch, batch_idx):
        print("** training **" * 2)
        x, y = batch
        y_hat = self.forward(x)
        print("train : " , np.shape(y_hat))
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss', avg_loss}
        #self.log("avg_val_loss", avg_loss)
        return {'avg_val_loss' : avg_loss, 'log' : tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-3)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser])
        print("** static on ** " * 2)
        parser.add_argument('--n_classes', type=int, default=10)
        return parser


class UnetDataModule(pl.LightningDataModule):
    def __init__(self, img_dir: str, batch_size = 4):
        super().__init__()
        
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = A.Compose([
            A.Normalize(),
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
    
    def prepare_data(self):
        print("** prepare data ** " * 2)
        train_data = CityDataset(self.img_dir + '/train', transform = self.transform)
        self.test_data = CityDataset(self.img_dir + '/val', transform = self.transform)
        train_data_size = int(len(train_data) * 0.8)
        print(train_data_size)
        val_data_size = len(train_data) - train_data_size
        print(val_data_size)
        print(len(self.test_data))
        self.train_data, self.val_data = random_split(train_data, [train_data_size, val_data_size])
        print("-" * 20)
        
    def train_dataloader(self):
        print("** train dataloader ** " * 2)
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
