import os
from matplotlib import transforms
import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from dataset import CityDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class UnetModel(pl.LightningModule):
    def __init__(self, params):
        super(UnetModel, self).__init__()
        self.n_classes = params.n_classes

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

        class Encoder(nn.Module): 
            def __init__(self, in_channels, out_channels):
                super(Encoder, self).__init__()
                self.encode = convBlockx2(in_channels, out_channels)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                x = self.encode(x) 
                x = self.pool(x)
                
                return x

        class Decoder(nn.Module): 
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
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        mid = self.mid(enc4)
        dec4 = self.dec4(torch.concat([mid, enc4], dim=1))
        dec3 = self.dec3(torch.concat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.concat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.concat([dec2, enc1], dim=1))
        final = self.final(dec1)

        return final

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
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


class UnetDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, batch_size = 32):
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
        train_data = CityDataset(self.img_dir + '/train', transform = self.transform)
        self.test_data = CityDataset(self.img_dir + '/val', transform = self.transform)

        train_data_size = int(len(train_data) * 0.8)
        val_data_size = len(train_data) - train_data_size
        print(train_data_size)
        self.train_data, self.val_data = data.random_split(train_data, [train_data_size, val_data_size])

    def train_dataloader(self):
        return data.DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(self.test_data, batch_size=self.batch_size)
