# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:40:48 2024

@author: USER
"""



import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

    
class TinyImageNet(Dataset):
    def __init__(self, train = True, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.num_classes = 200
        if train == True:
            dataset = load_dataset('Maysee/tiny-imagenet', split='train')
        else:
            dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
        self.data = dataset.with_format("torch")
        
    def __getitem__(self, index):
        
        img = self.data[index]["image"]
        label = self.data[index]["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
