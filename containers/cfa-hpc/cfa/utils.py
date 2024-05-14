# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from extra_datasets import TinyImageNet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot_encode(label_list, num_classes):
    
    encoded = torch.FloatTensor([[0 for i in range(num_classes)] for j in range(len(label_list))])
    for i in range(len(label_list)):
        encoded[i, label_list[i]] = 1

    return encoded


def get_average_of_min_20_percent(x):   # x will be a list
    x.sort()
    cutoff = int(len(x)*0.2)
    x_min_20 = x[:cutoff]
    
    return sum(x_min_20)/len(x_min_20)

def pgd_loss(model, x, y, eps, alpha, n_iters, normalize):
    delta = torch.zeros_like(x).to(x.device)
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0-x, 1-x)
    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(normalize(x+delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()

    robust_output = model(normalize(x+delta))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(robust_output, y)
    
    return loss, robust_output.clone().detach()

def cw_pgd_loss(model, x, y, eps, alpha, n_iters, normalize):
    delta = torch.zeros_like(x).to(x.device)
    base = torch.ones_like(x).to(x.device)
    for sample in range(len(x)):
        base[sample] *= eps[sample]
    eps = base.clone().to(x.device)

    delta = (torch.rand_like(delta) - 0.5) * 2
    delta.to(x.device)
    delta = delta * eps  # remargin

    delta = torch.clamp(delta, 0-x, 1-x)
    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(normalize(x+delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()
    delta.detach()
    robust_output = model(normalize(x+delta))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(robust_output, y)
    return loss, robust_output.clone().detach()

def fgsm_loss(model, x, y, eps, normalize):
    x = x.clone().detach().to(device)
    x.requires_grad = True
    output = model(normalize(x))
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    x_adv = x + eps * grad.sign()
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    robust_output = model(normalize(x_adv))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(robust_output, y)
    
    return loss, robust_output.clone().detach()

def cw_fgsm_loss(model, x, y, eps, normalize):
    base = torch.ones_like(x).to(x.device)
    for sample in range(len(x)):
        base[sample] *= eps[sample]
    eps = base.clone().to(x.device)
    x = x.clone().detach().to(device)
    x.requires_grad = True
    output = model(normalize(x))
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    x_adv = x + eps * grad.sign()
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    robust_output = model(normalize(x_adv))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(robust_output, y)
    
    return loss, robust_output.clone().detach()
    
def TRADES_loss(model, original_imgs, labels, normalize, epsilon = 8/255, beta = 6, step_size=0.003, num_steps=10, 
               ccr = "False", ccm= "False", eps_by_class = None, beta_by_class = None):  
    
    ### This function is mostly taken from https://github.com/PKU-ML/CFA/blob/main/attack.py
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(original_imgs)
    # generate adversarial example
    
    
    adv_imgs = original_imgs.detach() + 0.001 * torch.randn(original_imgs.shape).to(original_imgs.device).detach()
    
    for i in range(num_steps):
        adv_imgs.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(normalize(adv_imgs)), dim=1),
                                    F.softmax(model(normalize(original_imgs)), dim=1))
        grad = torch.autograd.grad(loss_kl, [adv_imgs])[0]
        adv_imgs = adv_imgs.detach() + step_size * torch.sign(grad.detach())
        if ccm == "True":
            batch_eps = torch.tensor([eps_by_class[int(label)] for label in labels])
            adv_imgs =  torch.clamp(adv_imgs, min = original_imgs - batch_eps, max = original_imgs + batch_eps)
        else:
            adv_imgs =  torch.clamp(adv_imgs, min = original_imgs - epsilon, max = original_imgs + epsilon)
        adv_imgs = torch.clamp(adv_imgs, min = 0, max = 1)

    model.train()

    adv_imgs = Variable(torch.clamp(adv_imgs, min = 0, max= 1), requires_grad=False)
    # calculate robust loss
    original_preds = model(normalize(original_imgs))
    
    loss_natural = F.cross_entropy(original_preds, labels, reduction='none') / batch_size

    cw_criterion = nn.KLDivLoss(reduction='none')
    
    adv_preds = model(normalize(adv_imgs))
    loss_robust = cw_criterion(F.log_softmax(adv_preds, dim=1), F.softmax(original_preds, dim=1)) / batch_size
    
    if ccr == "True":
        batch_beta = torch.tensor([beta_by_class[int(label)] for label in labels])
        assert len(batch_beta) == len(loss_robust)   
                                        
    print(batch_beta.shape, loss_natural.shape, loss_robust.shape)
    
    if ccr == "True":
        loss = ((1-batch_beta) * loss_natural + batch_beta * loss_robust).sum()
    else:
        loss = (loss_natural + beta * loss_robust).sum()
   

    return loss, adv_preds.clone().detach()
   
def weight_average(model, new_model, decay_rate, init=False): 
    ### This function is taken from https://github.com/PKU-ML/CFA/blob/main/utils.py
    
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    
    if init == True:
        decay_rate = 0
    
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict) 
    
    return 

def validation(model, valid_loader, normalize, attack, num_classes = 10):

    model.eval()
    ValidLoss = 0
    samples_by_class = [0 for i in range(num_classes)]
    clean_corrects_by_class = [0 for i in range(num_classes)]
    adv_corrects_by_class = [0 for i in range(num_classes)]
    
    loss_func=nn.CrossEntropyLoss()
    
    for j,(images,labels) in enumerate(valid_loader):
        images=images.to(device)
        labels=labels.to(device)
        adv_images = attack(images, labels)
        clean_predictions = model(normalize(images))
        _,clean_predicted=torch.max(clean_predictions,1)
        
        loss=loss_func(clean_predictions,labels)
        
        adv_predictions = model(normalize(adv_images))
       
        _,adv_predicted=torch.max(adv_predictions,1)
        
        for prediction, label in zip(list(clean_predicted), list(labels)):
            if prediction == label: 
                clean_corrects_by_class[int(label)] += 1
            samples_by_class[int(label)] += 1
        
        for prediction, label in zip(list(adv_predicted), list(labels)):
            if prediction == label: 
                adv_corrects_by_class[int(label)] += 1
            
        ValidLoss += loss.item()*labels.size(0)
    
    clean_accuracies_by_class = [correct/total for correct,total in zip(clean_corrects_by_class, samples_by_class)]
    adv_accuracies_by_class = [correct/total for correct,total in zip(adv_corrects_by_class, samples_by_class)]
    ValidLoss = ValidLoss/sum(samples_by_class)
        
    return clean_accuracies_by_class, adv_accuracies_by_class, ValidLoss    

def calculate_test_accs(model, test_loader, normalize, pgd_attack, fgsm_attack, num_classes = 10):
    
    
    model.eval()
    samples_by_class = [0 for i in range(num_classes)]
    clean_corrects_by_class = [0 for i in range(num_classes)]
    pgd_corrects_by_class = [0 for i in range(num_classes)]
    fgsm_corrects_by_class = [0 for i in range(num_classes)]
    
    for j,(images,labels) in enumerate(test_loader):
        images=images.to(device)
        labels=labels.to(device)
        pgd_adv_images = pgd_attack(images, labels)
        fgsm_adv_images = fgsm_attack(images, labels)
        
        clean_predictions = model(normalize(images))
        _,clean_predicted=torch.max(clean_predictions,1)
    
        pgd_predictions = model(normalize(pgd_adv_images))
        _,pgd_predicted=torch.max(pgd_predictions,1)
        
        fgsm_predictions = model(normalize(fgsm_adv_images))
        _,fgsm_predicted=torch.max(fgsm_predictions,1)
        
        for prediction, label in zip(list(clean_predicted), list(labels)):
            if prediction == label: 
                clean_corrects_by_class[int(label)] += 1
            samples_by_class[int(label)] += 1
        
        for prediction, label in zip(list(pgd_predicted), list(labels)):
            if prediction == label: 
                pgd_corrects_by_class[int(label)] += 1

        for prediction, label in zip(list(fgsm_predicted), list(labels)):
            if prediction == label: 
                fgsm_corrects_by_class[int(label)] += 1
            
            
    
    clean_accuracies_by_class = [correct/total for correct,total in zip(clean_corrects_by_class, samples_by_class)]
    pgd_accuracies_by_class = [correct/total for correct,total in zip(pgd_corrects_by_class, samples_by_class)]
    fgsm_accuracies_by_class = [correct/total for correct,total in zip(fgsm_corrects_by_class, samples_by_class)]
        
    return clean_accuracies_by_class, pgd_accuracies_by_class, fgsm_accuracies_by_class    


def get_loaders(dataset_name = "cifar10", valid_size = 0.02):
    cifar10_transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])

    cifar10_transform_valid = transforms.Compose([transforms.ToTensor()])

    tiny_imagenet_transform_train = transforms.Compose([transforms.RandomCrop(56), transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()])

    tiny_imagenet_transform_valid = transforms.Compose([transforms.CenterCrop(56), transforms.ToTensor()])


    cifar10_train = datasets.CIFAR10(root="./data", train = True, download = True, transform=cifar10_transform_train)
    cifar10_valid = datasets.CIFAR10(root="./data", train = True, download = True, transform=cifar10_transform_valid)
    cifar10_test = datasets.CIFAR10(root="./data", train = False, download = True, transform=cifar10_transform_valid)
    tiny_imagenet_train = TinyImageNet(train = True, transform = tiny_imagenet_transform_train)
    tiny_imagenet_valid = TinyImageNet(train = True, transform = tiny_imagenet_transform_valid)
    tiny_imagenet_test = TinyImageNet(train = False, transform = tiny_imagenet_transform_valid)
    
    
    rng = np.random.default_rng(42)
    train_indices = []
    valid_indices = []
    
    if dataset_name == "cifar10":
        num_train = len(cifar10_train)
        indices = list(rng.permutation(num_train))
        num_classes = 10
        valid_amount = int(np.floor(valid_size * num_train)/num_classes)
        class_indices = [[] for i in range(num_classes)]
        labels = cifar10_train.targets
        for index in indices:
            class_indices[labels[index]].append(index)
            
        for i in range(num_classes):
            valid_indices += class_indices[i][:valid_amount]
            train_indices += class_indices[i][valid_amount:]
       
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        train_loader = DataLoader(cifar10_train, batch_size = 128, sampler = train_sampler)
        valid_loader = DataLoader(cifar10_valid, batch_size = 128, sampler = valid_sampler)
        test_loader = DataLoader(cifar10_test, batch_size = 128, shuffle = False)
    elif dataset_name == "tiny_imagenet":
        num_train = len(tiny_imagenet_train)
        indices = list(rng.permutation(num_train))
        num_classes = 200
        valid_amount = int(np.floor(valid_size * num_train)/num_classes)
        class_indices = [[] for i in range(num_classes)]
        labels = cifar10_train.targets
        for index in indices:
            class_indices[labels[index]].append(index)
            
        for i in range(num_classes):
            valid_indices += class_indices[i][:valid_amount]
            train_indices += class_indices[i][valid_amount:]
       
        train_loader = DataLoader(tiny_imagenet_train, batch_size = 128, sampler = train_sampler)
        valid_loader = DataLoader(tiny_imagenet_valid, batch_size = 128, shuffle = valid_sampler)
        test_loader = DataLoader(tiny_imagenet_test, batch_size = 128, shuffle = False)
    
    return train_loader, valid_loader, test_loader

