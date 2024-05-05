# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import torchvision.models as models
from model import ResNet18, PreActResNet18
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from extra_datasets import TinyImageNet
from utils import pgd_attack, calculate_valid_acc
import math

def train_default(dataset_name = "cifar10", batch_size = 128, lr = 0.1, momentum = 0.9, weight_decay = 5e-4, epoch_number = 200):
    
    if(dataset_name == "cifar10"):
        
        train_loader = DataLoader(cifar10_train, batch_size = batch_size, sampler = cifar10_train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size = batch_size, sampler = cifar10_valid_sampler)
        model = PreActResNet18(dataset_name = dataset_name).to(device)
        normalize = cifar10_normalize
        denormalize = cifar10_denormalize
    if(dataset_name == "tiny-imagenet"):
        
        train_loader = DataLoader(tiny_imagenet_train, batch_size = batch_size, sampler = tiny_imagenet_train_sampler)
        valid_loader = DataLoader(tiny_imagenet_valid, batch_size = batch_size, sampler = tiny_imagenet_valid_sampler)
        model = ResNet18(dataset_name = dataset_name).to(device)
        normalize = tiny_imagenet_normalize
        denormalize = tiny_imagenet_denormalize
        
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epoch_number/2),int(3/4*epoch_number)], gamma=0.1)
    loss_func=nn.CrossEntropyLoss()
    best_valid_accuracy = 0
    
    for epoch in range(epoch_number):
        AverageLoss=0
        model.train()
        total_samples = 0
        for i,(images,labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            adv_images = pgd_attack(images, labels, model, loss_func, normalize, denormalize, epsilon = 8/255, num_steps = 3, step_size = 2/255)
            predictions = model(normalize(adv_images))
            #predictions = model(images)
            loss=loss_func(predictions,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            AverageLoss+=loss.item()*images.size(0)
            total_samples += images.size(0)
        if(epoch%10 == 1):
            valid_accuracy, validloss = calculate_valid_acc(model, valid_loader)
            if(valid_accuracy > best_valid_accuracy):
                best_valid_accuracy = valid_accuracy
        
        scheduler.step()
        print("Training loss is {loss} at the end of epoch  {epoch}".format(loss = AverageLoss/total_samples, epoch = epoch +1))
        print(total_samples)
    return best_valid_accuracy



device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar10_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

tiny_imagenet_normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])

def cifar10_denormalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).to(device)
    return x*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def tiny_imagenet_denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return x*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

cifar10_transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), cifar10_normalize,])

cifar10_transform_valid = transforms.Compose([transforms.ToTensor(), cifar10_normalize])


tiny_imagenet_transform_train = transforms.Compose([transforms.RandomCrop(56), transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(), tiny_imagenet_normalize,])

tiny_imagenet_transform_valid = transforms.Compose([transforms.CenterCrop(56), transforms.ToTensor(), tiny_imagenet_normalize])


cifar10_train = datasets.CIFAR10(root="./data", train = True, download = True, transform=cifar10_transform_train)
cifar10_valid = datasets.CIFAR10(root="./data", train = True, download = True, transform=cifar10_transform_valid)
tiny_imagenet_train = TinyImageNet(train = True, transform = tiny_imagenet_transform_train)
tiny_imagenet_valid = TinyImageNet(train = True, transform = tiny_imagenet_transform_valid)


def get_samplers_for_valid_and_train(dataset_name = "cifar10", valid_size = 0.1):
    valid_size = valid_size
    
    if dataset_name == "cifar10":
        num_train = len(cifar10_train)
    elif dataset_name == "tiny_imagenet":
        num_train = len(tiny_imagenet_train)
    
    rng = np.random.default_rng(42)
    indices = list(rng.permutation(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    return train_sampler, valid_sampler

cifar10_train_sampler, cifar10_valid_sampler = get_samplers_for_valid_and_train("cifar10", valid_size = 0.1)
tiny_imagenet_train_sampler, tiny_imagenet_valid_sampler = get_samplers_for_valid_and_train("tiny_imagenet", valid_size = 0.1)





train_default(epoch_number = 2, dataset_name = "cifar10", lr = 0.01, batch_size = 128)
