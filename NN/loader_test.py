import os
import torch
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import Dataset

class IsingDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = torch.load(label_dir)
        self.img_data = torch.load(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label