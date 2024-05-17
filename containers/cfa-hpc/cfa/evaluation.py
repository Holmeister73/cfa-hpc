# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchattacks

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def final_evaluation(model, test_loader, mean, std, normalize, dataset_name = "cifar10"):
    if dataset_name == "cifar10":
        num_classes = 10
        epsilon = 8/255
    elif dataset_name == "tiny_imagenet":
        num_classes = 200
        epsilon = 4/255
    atk1 = torchattacks.APGD(model, eps = epsilon, norm = "Linf", loss = "ce", n_restarts = 1)
    atk2 = torchattacks.APGDT(model, eps = epsilon, norm = "Linf", n_classes = num_classes, n_restarts = 1)
    atk3 = torchattacks.FAB(model, eps = epsilon, norm = "Linf", multi_targeted = True, n_classes = num_classes, n_restarts = 1)
    atk4 = torchattacks.Square(model, eps = epsilon, norm = "Linf", n_queries = 5000, n_restarts = 1)
    attacks = [atk1, atk2, atk3, atk4]
    for attack in attacks:
        attack.set_normalization_used(mean, std)
        attack._set_normalization_applied(False)
        
    multiattack = torchattacks.MultiAttack(attacks)
    autoattack = torchattacks.AutoAttack(model)
    autoattack._autoattack = multiattack
    #autoattack.normalization_used = {"mean": mean, "std": std}
    #autoattack._set_normalization_applied(False)
    #autoattack.set_normalization_used(mean, std)
    
    model.eval()
    samples_by_class = [0 for i in range(num_classes)]
    clean_corrects_by_class = [0 for i in range(num_classes)]
    robust_corrects_by_class = [0 for i in range(num_classes)]
    
    for j,(images,labels) in enumerate(test_loader):
        images=images.to(device)
        labels=labels.to(device)
        adv_images = autoattack(images, labels)

        clean_predictions = model(normalize(images))
        _,clean_predicted=torch.max(clean_predictions,1)
        
        robust_predictions = model(normalize(adv_images))
       
        _,robust_predicted=torch.max(robust_predictions,1)
        
        for prediction, label in zip(list(clean_predicted), list(labels)):
            if prediction == label: 
                clean_corrects_by_class[int(label)] += 1
            samples_by_class[int(label)] += 1
        
        for prediction, label in zip(list(robust_predicted), list(labels)):
            if prediction == label: 
                robust_corrects_by_class[int(label)] += 1

            
    
    clean_accuracies_by_class = [correct/total for correct,total in zip(clean_corrects_by_class, samples_by_class)]
    robust_accuracies_by_class = [correct/total for correct,total in zip(robust_corrects_by_class, samples_by_class)]
        
    return clean_accuracies_by_class, robust_accuracies_by_class 




