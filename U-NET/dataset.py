import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from PIL import Image
import albumentations
from sklearn.cluster import KMeans

class CityDataset(Dataset):
    def __init__(self, img_dir, n_classes, transform = None):
        super(CityDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform  = transform
        self.n_classes = n_classes
        # output label define

        num_items = 1000
        color_array = np.random.choice(range(256), 3 * num_items).reshape(-1, 3)

        self.label_model = KMeans(n_clusters=n_classes)
        self.label_model.fit(color_array)

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
        
        label_class = torch.Tensor(self.label_define(label)).long()
            
        return img, label_class

    def split_image(self, img):
        img = np.array(img)
        img, label = img[ : , :256 , : ], img[ : , 256: , :]
        return img, label

    def label_define(self, label):
        return self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
