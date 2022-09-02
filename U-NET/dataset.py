import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class UnetDataset(Dataset):
    def __init__(self, img_dir, transform = None):
        super(UnetDataset, self).__init__()
        self.img_list = os.listdir(img_dir)
        self.transform  = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_path = os.path.join(self.img_list, img_path)
        img = np.array(Image.open(img_path), )
        img, label = self.split_image(img)

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def split_image(self, img):
        img = np.array(img)
        img = img / 255.0
        img, label = img[ : , :256 , : ], img[ : , 256: , :]
        return img, label