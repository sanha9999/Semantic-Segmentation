import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class UnetDataset(Dataset):
    def __init__(self, img_dir):
        super(UnetDataset, self).__init__()
        self.img_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_path = os.path.join(self.img_list, img_path)
        img = np.array(Image.open(img_path))
        img, label = self 

    def split_image(self, img):
        img = np.array(img)
        img, label = img[ : , :256 , : ], img[ : , 256: , :]   