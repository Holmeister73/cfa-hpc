# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchattacks

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def final_evaluation(model, test_loader, mean, std, normalize, num_classes = 10):
    
    autoattack = torchattacks.AutoAttack(model, norm='Linf', eps = 8/255, n_classes = num_classes)
    autoattack.normalization_used = {"mean": mean, "std": std}
    autoattack._set_normalization_applied(False)
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




