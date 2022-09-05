import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from PIL import Image
import albumentations

class CityDataset(Dataset):
    def __init__(self, img_dir, transform = None):
        super(CityDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform  = transform 

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_path)
        img = np.array(Image.open(img_path).convert('RGB'))
        img, label = self.split_image(img)

        if self.transform:
            augmented = self.transform(image = img, mask = label)
            img = augmented['image']
            label = augmented['mask']
            
        return img, label

    def split_image(self, img):
        img = np.array(img)
        #img = img / 255.0
        img, label = img[ : , :256 , : ], img[ : , 256: , :]
        return img, label