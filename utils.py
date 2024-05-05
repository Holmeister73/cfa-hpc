# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:39:22 2024

@author: USER
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encode(label_list, num_classes):
    
    encoded = torch.FloatTensor([[0 for i in range(num_classes)] for j in range(len(label_list))])
    for i in range(len(label_list)):
        encoded[i, label_list[i]] = 1

    return encoded

def pgd_attack(original_img, label, model, loss_func, normalize, denormalize, epsilon = 8/255, num_steps = 10, step_size = 2/255):
    original_img = denormalize(original_img)
    x = original_img + 2*epsilon*torch.rand(original_img.shape).to(device) - epsilon*torch.ones_like(original_img).to(device)
    x = torch.clamp(x, 0, 1)
    x.requires_grad = True
    for i in range(num_steps):
        pred = model(normalize(x))
        loss = loss_func(pred, label)
        loss.backward()
        grad = x.grad.detach()
        x_data = x + step_size*torch.sign(grad)
        x_data = torch.clamp(x_data, min = original_img-epsilon, max = original_img + epsilon)
        x_data = torch.clamp(x_data, 0, 1)
        x.data = x_data
        x.grad.zero_()
        
    return x.detach()        



def soft_cross_entropy(output, target, soft_labels):    
    
    target_prob = torch.zeros_like(output)
    batch = output.shape[0]
    for k in range(batch):
            target_prob[k] = soft_labels[int(target[k])]
            
    log_like = -torch.nn.functional.log_softmax(output, dim=1)
    loss = torch.sum(torch.mul(log_like, target_prob)) / batch 
    return loss

def cross_entropy_without_softmax(predictions, labels, num_classes):
    
    batch = predictions.shape[0]
    log_predictions = -torch.log(predictions)
    labels = F.one_hot(labels, num_classes).float()
    loss = torch.sum(torch.mul(log_predictions,labels)) / batch
    
    return loss

def manual_cross_entropy(predictions, labels):
    batch = predictions.shape[0]
    log_softmax_predictions = -torch.nn.functional.log_softmax(predictions, dim=1)
    loss = torch.sum(torch.mul(log_softmax_predictions,labels)) / batch
    
    return loss

def entropy(prediction):
    batch = prediction.shape[0]
    prediction_log_softmax = -torch.nn.functional.log_softmax(prediction, dim=1)
    prediction_softmax = torch.nn.functional.softmax(prediction, dim = 1)
    loss = torch.sum(torch.mul(prediction_log_softmax, prediction_softmax)) / batch
    
    return loss

def kl_loss(predictions, log_labels):
    batch = predictions.shape[0]
    predictions_softmax = torch.nn.functional.softmax(predictions, dim = 1)
    log_softmax_predictions = torch.nn.functional.log_softmax(predictions, dim=1)
    loss = torch.sum(torch.mul(predictions_softmax,(log_softmax_predictions-log_labels))) / batch
    
    return loss

def calculate_valid_acc(network, valid_loader):
    
    with torch.no_grad():
        ValidLoss = 0
        network.eval()
        samples=0
        correct=0
        total = 0
        loss_func=nn.CrossEntropyLoss()
        for j,(images,labels) in enumerate(valid_loader):
            images=images.to(device)
            labels=labels.to(device)
            
            predictions = network(images)
            
            loss=loss_func(predictions,labels)
            
            _,predicted=torch.max(predictions,1)
            samples+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            ValidLoss+= loss.item()*labels.size(0)
            total += labels.size(0)
    
    acc=correct/samples
    
    return acc, ValidLoss/total      


