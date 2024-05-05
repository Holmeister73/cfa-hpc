# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:47:30 2024

@author: USER
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from extra_datasets import Cub200Dataset, Dataset_with_Indices



def evaluate(model, dataset = "cifar10", method = "normal"):
    
    if(dataset == "cifar10"):
        test_loader=torch.utils.data.DataLoader(cifar10_test,batch_size=batch_size,shuffle=False)
            
    if(dataset == "cifar100"):
        test_loader=torch.utils.data.DataLoader(cifar100_test,batch_size=batch_size,shuffle=False)

    with torch.no_grad():
       model.eval()
       samples=0
       correct=0
      
       for j,(images,labels) in enumerate(test_loader):
           images = images.to(device)
           labels = labels.to(device)
           predictions, embeddings = model(images)
           _,predicted = torch.max(predictions,1)
           
           samples += labels.size(0)
           correct += (predicted==labels).sum().item()
               
       acc = correct/samples
           
    return acc

def fgsm_attack(image, epsilon, data_grad):   # This code is from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
   
    return perturbed_image

# restores the tensors to their original scale
def denormalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]): #This code is from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
   

    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def fgsm_evaluate( model, dataset = "cifar10", method = "normal", epsilon = 0.1 ): #This code is from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

    
    correct = 0
    adv_examples = []
    loss_func = nn.CrossEntropyLoss()
    if(dataset == "cifar10"):
        test_loader=torch.utils.data.DataLoader(cifar10_test,batch_size=1,shuffle=False)
        normalize = cifar10_normalize
        mean = cifar10_mean
        std = cifar10_std
        if(method == "pencil"):
            pencil_test = Dataset_with_Indices(cifar10_test)
            test_loader = torch.utils.data.DataLoader(pencil_test,batch_size=1,shuffle=False)   
            
    if(dataset == "cifar100"):
        test_loader=torch.utils.data.DataLoader(cifar100_test,batch_size=1,shuffle=False)
        normalize = cifar100_normalize
        mean = cifar100_mean
        std = cifar100_std
        if(method == "pencil"):
            pencil_test = Dataset_with_Indices(cifar100_test)
            test_loader = torch.utils.data.DataLoader(pencil_test,batch_size=1,shuffle=False)   
   
    for i,(image, label) in enumerate(test_loader):

        image, label = image.to(device), label.to(device)

        image.requires_grad = True

        prediction, embedding = model(image)
        _,predicted = torch.max(prediction,1)

        if predicted.item() != label.item():
            continue

        loss = loss_func(prediction, label)

        model.zero_grad()

        loss.backward()

        image_grad = image.grad.data

        denormalized_image = denormalize(image, mean = mean, std = std)

        perturbed_image = fgsm_attack(denormalized_image, epsilon, image_grad)

        perturbed_image_normalized = normalize(perturbed_image)

        new_prediction, embedding = model(perturbed_image_normalized)

      
        _,new_predicted = torch.max(new_prediction,1) 
        
        if new_predicted.item() == label.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (predicted.item(), new_predicted.item(), adv_ex) )
        else:
            
            if len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (predicted.item(), new_predicted.item(), adv_ex) )

    
    final_acc = correct/len(test_loader)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    return final_acc, adv_examples

def ECE (model, dataset = "cifar10", num_bins = 15, method = "normal"):
    
    confidences = [[] for i in range(num_bins)]
    corrects = [0 for i in range(num_bins)]
    num_of_predictions = [0 for i in range(num_bins)]
    
    if(dataset == "cifar10"):
        test_loader=torch.utils.data.DataLoader(cifar10_test,batch_size=1,shuffle=False)
    if(dataset == "cifar100"):
        test_loader=torch.utils.data.DataLoader(cifar100_test,batch_size=1,shuffle=False)
        
    with torch.no_grad():
       model.eval()
      
       for j,(images,labels) in enumerate(test_loader):
           images = images.to(device)
           labels = labels.to(device)
           predictions, embeddings = model(images)
            
           softmax_predictions = F.softmax(predictions, dim = 1)
           confidence,predicted = torch.max(softmax_predictions,1)
           bin_number = int(torch.floor(confidence*num_bins))
           confidences[bin_number].append(confidence)
           num_of_predictions[bin_number] += 1
           
           if(labels.item() == predicted.item()):
               corrects[bin_number] += 1
                   
           
           
    
    ece = 0          
    for i in range(num_bins):
        if num_of_predictions[i] == 0:
            continue
        accuracy = corrects[i]/num_of_predictions[i]
        average_confidence = (sum(confidences[i]) / len(confidences[i])).item()
        
        ece += (num_of_predictions[i]/len(test_loader))*abs(accuracy-average_confidence)
           
    return ece

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std= [0.2470, 0.2435, 0.2616]
cifar100_mean = [0.5071, 0.4865, 0.4409]
cifar100_std = [0.2673, 0.2564, 0.2762]
cub200_mean = [0.4856, 0.4994, 0.4324]
cub200_std = [0.2264, 0.2218, 0.2606]

cifar10_normalize = transforms.Normalize(mean=cifar10_mean,
                                         std=cifar10_std)

cifar100_normalize = transforms.Normalize(mean=cifar100_mean,
                                         std=cifar100_std)

cub200_normalize = transforms.Normalize(mean = cub200_mean,
                                         std = cub200_std)
cifar10_transform = transforms.Compose([
           transforms.ToTensor(),
           cifar10_normalize
       ])

cifar100_transform = transforms.Compose([
           transforms.ToTensor(),
           cifar100_normalize
       ])

cub200_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        cub200_normalize
    ])



batch_size = 128


cifar10_test = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=cifar10_transform)
cifar100_test = torchvision.datasets.CIFAR100(root="./data",train=False,download=True,transform=cifar100_transform)


