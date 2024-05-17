# -*- coding: utf-8 -*-


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
            
        self.targets = list(dataset["label"])
        #self.data = dataset.with_format("torch")
        self.data = dataset
        
    def __getitem__(self, index):
        index = int(index)
        img = self.data[index]["image"]
        
        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
            
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        return img, label

    def __len__(self):
        return len(self.data)
