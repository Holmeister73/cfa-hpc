# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import ResNet18, PreActResNet18
from utils import validation, calculate_test_accs, get_loaders, TRADES_loss, weight_average, get_average_of_min_20_percent
from utils import pgd_loss, fgsm_loss, cw_pgd_loss, cw_fgsm_loss
from evaluation import final_evaluation
import math
import argparse
import torchattacks
import pandas as pd
import datasets
import logging

logging.basicConfig(level = logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

parser=argparse.ArgumentParser()

parser.add_argument("--dataset_name", help = """Name of the dataset, it can either be cifar10 or tiny_imagenet""", type = str, default = "cifar10")
                    
                    
parser.add_argument("--model_name", help="""Model architechture it can either be ResNet18 or PreActResNet18""",
                    type = str, default = "PreActResNet18")

parser.add_argument("--num_epochs", help="""Number of epochs, validation evaluation is done at each epoch""",
                    type = int, default = 200)

parser.add_argument("--batch_size", help="""Batch size for training and evaluation""", type = int, default = 128)

parser.add_argument("--initial_lr", help="""Initial learning rate for sgd with momentum""",
                    type = float, default = 0.1)

parser.add_argument("--momentum", help="""Momentum for sgd""", type = float, default = 0.9)

parser.add_argument("--weight_decay", help="""Weight of L2 regularization""", type = float, default = 5e-4)

parser.add_argument("--training_type", help="""Type of adversarial training. It can either be vanilla or TRADES""", type = str, default = "vanilla")

parser.add_argument("--attack_type", help = """Type of attack that will be used for vanilla adversarial training,
                    it can be pgd or fgsm""", type = str, default = "pgd")
                    
parser.add_argument("--epsilon", help="""Attack strength for pgd. It will be divided by 255 in the code.""", type = int, default = 8)

parser.add_argument("--step_size", help="""Step size for pgd attack. It will be divided by 255 in the code.""", type = int, default = 2)

parser.add_argument("--attack_steps", help = "Amount of steps for pgd attack and TRADES loss", type = int, default = 10)

parser.add_argument("--beta", help = "Beta hyperparameter for TRADES", type = float, default = 6)

parser.add_argument("--ccm", help="""Whether to use ccm, default is false""", type = str,  default = "False")

parser.add_argument("--ccr", help="""Whether to use ccr, default is false""", type = str,  default = "False")

parser.add_argument("--weight_average_type", help="""What type of weight averaging to use, options are ema, fawa, none""", type = str,  default = "none")

parser.add_argument('--lambda_1', help = "Value of lambda-1 which is used in ccm", default=0.5, type=float)

parser.add_argument('--lambda_2', help = "Value of lambda-2 which is used in ccr", default=0.5, type=float)
    
parser.add_argument('--decay-rate', help = "Decay rate of ema and fawa", default=0.85 ,type=float)

parser.add_argument('--threshold', help = "Threshold used in ema and fawa", default=0.2, type=float)  #0.2 for CIFRA10 0.04 for TinyImageNet
    
parser.add_argument("--hf_token_hub", help="""Token for huggingface""",  default = None)

parser.add_argument("--detailed_statistics", help="""Whether to export detailed statistics to hf hub""", type = str,  default = "False")


args=parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
lr = args.initial_lr
epoch_number = args.num_epochs
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
training_type = args.training_type
attack_type = args.attack_type
epsilon = args.epsilon/255
step_size = args.step_size/255
num_steps = args.attack_steps
beta = args.beta
ccm = args.ccm
ccr = args.ccr
weight_average_type = args.weight_average_type
lambda_1 = args.lambda_1
lambda_2 = args.lambda_2
decay_rate = args.decay_rate
threshold = args.threshold

hf_token = args.hf_token_hub
detailed_statistics = args.detailed_statistics

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


cifar10_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

tiny_imagenet_normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])

train_loader, valid_loader, test_loader = get_loaders(dataset_name = dataset_name, valid_size = 0.02)   

if model_name == "PreActResNet18":
    model = PreActResNet18(dataset_name = dataset_name).to(device)
    best_model = PreActResNet18(dataset_name = dataset_name).to(device)
    best_model.eval()
    if weight_average_type == "ema":
        ema_model = PreActResNet18(dataset_name = dataset_name).to(device)
        ema_model.eval()
    
    if weight_average_type == "fawa":
        fawa_model = PreActResNet18(dataset_name = dataset_name).to(device)
        fawa_model.eval()
else:
    model = ResNet18(dataset_name = dataset_name).to(device)
    best_model = ResNet18(dataset_name = dataset_name).to(device)
    best_model.eval()
    if weight_average_type == "ema":
        ema_model = ResNet18(dataset_name = dataset_name).to(device)
        ema_model.eval()
        
    if weight_average_type == "fawa":
        fawa_model = ResNet18(dataset_name = dataset_name).to(device)
        fawa_model.eval()

#eval_pgd_attack = torchattacks.PGD(model, eps = epsilon, alpha = step_size, steps = num_steps, random_start = False)
#eval_fgsm_attack = torchattacks.FGSM(model, eps = epsilon)

if(dataset_name == "cifar10"):
    normalize = cifar10_normalize
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616] 
    #eval_pgd_attack.set_normalization_used(mean = mean , std = std)  
    #eval_fgsm_attack.set_normalization_used(mean = mean , std = std) 
if(dataset_name == "tiny_imagenet"):
    normalize = tiny_imagenet_normalize
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #eval_pgd_attack.set_normalization_used(mean = mean , std = std)  
    #eval_fgsm_attack.set_normalization_used(mean = mean , std = std)

    
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epoch_number/2),int(3/4*epoch_number)], gamma=0.1)
loss_func=nn.CrossEntropyLoss()

eps_by_class = [epsilon for i in range(num_classes)]
beta_by_class = [beta/(beta+1) for i in range(num_classes)]
classwise_test_pgd_accuracies_for_all_epochs = []
classwise_test_fgsm_accuracies_for_all_epochs = []
classwise_test_clean_accuracies_for_all_epochs = []
best_model_threshold = 0

for epoch in range(epoch_number):
    AverageLoss = 0
    model.train()
    total_samples = 0 
    train_robust_corrects_by_class = [0 for i in range(num_classes)]
    samples_by_class = [0 for i in range(num_classes)]    
    
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #batch_eps = torch.Tensor([eps_by_class[int(label)] for label in list(labels)])
        
        if training_type == "vanilla":
            if attack_type == "pgd":
                if ccm == "True":
                    #pgd_attack = torchattacks.PGD(model, eps = batch_eps, alpha = step_size, steps = num_steps)
                    loss, robust_output = cw_pgd_loss(model, images, labels, eps_by_class, step_size, num_steps, normalize)
                else:
                    #pgd_attack = torchattacks.PGD(model, eps = epsilon, alpha = step_size, steps = num_steps)
                    loss, robust_output = pgd_loss(model, images, labels, epsilon, step_size, num_steps, normalize)
                    
                #pgd_attack.set_normalization_used(mean = mean , std = std)
                
                #adv_images = pgd_attack(images, labels)
                
            elif attack_type == "fgsm":
                if ccm == "True":
                    #fgsm_attack = torchattacks.FGSM(model, eps = batch_eps)
                    loss, robust_output = cw_fgsm_loss(model, images, labels, eps_by_class, normalize)
                else:
                    #fgsm_attack = torchattacks.FGSM(model, eps = epsilon)
                    loss, robust_output = fgsm_loss(model, images, labels, epsilon, normalize)
                #fgsm_attack.set_normalization_used(mean = mean , std = std)
                #adv_images = fgsm_attack(images, labels)
                
            #predictions = model(normalize(adv_images))
            #loss=loss_func(predictions,labels)
            
            #_,predicted=torch.max(predictions,1)
            _,predicted = torch.max(robust_output,1)
            for prediction, label in zip(list(predicted), list(labels)):
                if prediction == label: 
                    train_robust_corrects_by_class[int(label)] += 1
                
                samples_by_class[int(label)] += 1
            
            
        elif training_type == "TRADES":
            loss, robust_output = TRADES_loss(model, images, labels, normalize, epsilon = epsilon, beta = beta, step_size = 0.003, num_steps = num_steps, 
                           ccr = ccr, ccm= ccm, eps_by_class = eps_by_class, beta_by_class = beta_by_class)
            
            _,predicted=torch.max(robust_output,1)
            
            for prediction, label in zip(list(predicted), list(labels)):
                if prediction == label: 
                    train_robust_corrects_by_class[int(label)] += 1
                
                samples_by_class[int(label)] += 1
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        AverageLoss+=loss.item()*images.size(0)
        total_samples += images.size(0)
    
    #if training_type == "vanilla":
    #    if attack_type == "pgd":    
            #eval_attack = eval_pgd_attack
    #        eval_attack = "pgd"
    #    elif attack_type == "fgsm":
            #eval_attack = eval_fgsm_attack
    #        eval_attack = fgsm_loss
    #elif training_type == "TRADES":
        #eval_attack = eval_pgd_attack
    #    eval_attack = pgd_loss
    eval_attack = "pgd"
    if attack_type == "fgsm":
      eval_attack = "fgsm"
    
    valid_clean_accuracies_by_class, valid_adv_accuracies_by_class, validloss = validation(model, valid_loader, normalize, attack = eval_attack, 
                                                                                           num_classes = num_classes)
   
    test_clean_accuracies_by_class, test_pgd_accuracies_by_class, test_fgsm_accuracies_by_class = calculate_test_accs(model, test_loader, normalize, num_classes = num_classes)
    
    train_robust_accuracies_by_class = [correct/sample for correct, sample in zip(train_robust_corrects_by_class, samples_by_class)]
    
    if ccm == "True" and epoch >= 9:
        eps_by_class = [epsilon*(lambda_1 + train_robust_acc) for train_robust_acc in train_robust_accuracies_by_class]    
    
    if ccr == "True" and epoch >= 9:
        beta_by_class = [beta*(lambda_2 + train_robust_acc)/(beta*(lambda_2 + train_robust_acc)+1) for  train_robust_acc in train_robust_accuracies_by_class]
    
    if weight_average_type == "ema":
        if epoch == 49:
            weight_average(ema_model, model, decay_rate, init = True)
            ema_test_clean_accuracies_by_class, ema_test_pgd_accuracies_by_class, ema_test_fgsm_accuracies_by_class = calculate_test_accs(ema_model, 
                                                           test_loader, normalize, num_classes = num_classes)
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = ema_test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = ema_test_pgd_accuracies_by_class
        elif epoch > 49:
            if dataset_name == "cifar10":
                if min(valid_adv_accuracies_by_class) >= threshold:
                    weight_average(ema_model, model, decay_rate, init = False)
            elif dataset_name == "tiny_imagenet":
                if get_average_of_min_20_percent(valid_adv_accuracies_by_class) >= threshold:
                    weight_average(ema_model, model, decay_rate, init = False)
            ema_test_clean_accuracies_by_class, ema_test_pgd_accuracies_by_class, ema_test_fgsm_accuracies_by_class = calculate_test_accs(ema_model, 
                                                           test_loader, normalize, num_classes = num_classes)
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = ema_test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = ema_test_pgd_accuracies_by_class
            if sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class) > best_model_threshold:
                weight_average(best_model, ema_model, decay_rate, init = True)
                best_model_threshold = sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class)
        else:
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = test_pgd_accuracies_by_class
    elif weight_average_type == "fawa":
        if epoch == 49:
            weight_average(fawa_model, model, decay_rate, init = True)
            fawa_test_clean_accuracies_by_class, fawa_test_pgd_accuracies_by_class, fawa_test_fgsm_accuracies_by_class = calculate_test_accs(fawa_model, 
                                                           test_loader, normalize, num_classes = num_classes)
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = fawa_test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = fawa_test_pgd_accuracies_by_class
        elif epoch > 49:
            if dataset_name == "cifar10":
                if min(valid_adv_accuracies_by_class) >= threshold:
                    weight_average(fawa_model, model, decay_rate, init = False)
            elif dataset_name == "tiny_imagenet":
                if get_average_of_min_20_percent(valid_adv_accuracies_by_class) >= threshold:
                    weight_average(fawa_model, model, decay_rate, init = False)
            
            fawa_test_clean_accuracies_by_class, fawa_test_pgd_accuracies_by_class, fawa_test_fgsm_accuracies_by_class = calculate_test_accs(fawa_model, 
                                                           test_loader, normalize,  num_classes = num_classes)
        
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = fawa_test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = fawa_test_pgd_accuracies_by_class
            if sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class) > best_model_threshold:
                weight_average(best_model, fawa_model, decay_rate, init = True)
                best_model_threshold = sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class)
        else:
            if attack_type == "fgsm":
                test_robust_accuracies_by_class = test_fgsm_accuracies_by_class
            else:
                test_robust_accuracies_by_class = test_pgd_accuracies_by_class
    else:
        if attack_type == "fgsm":
            test_robust_accuracies_by_class = test_fgsm_accuracies_by_class
        else:
            test_robust_accuracies_by_class = test_pgd_accuracies_by_class
        
        if sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class) > best_model_threshold:
            weight_average(best_model, model, decay_rate, init = True)
            best_model_threshold = sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class) + min(test_robust_accuracies_by_class)
    scheduler.step()
    logging.info("Training loss is {loss} at the end of epoch  {epoch}".format(loss = AverageLoss/total_samples, epoch = epoch + 1))
    logging.info("Average test robustness accuracy = {robust_acc}".format(robust_acc = sum(test_robust_accuracies_by_class)/len(test_robust_accuracies_by_class)))
    if weight_average_type == "ema":
        if epoch >= 49:
            classwise_test_pgd_accuracies_for_all_epochs.append(ema_test_pgd_accuracies_by_class)
            classwise_test_fgsm_accuracies_for_all_epochs.append(ema_test_fgsm_accuracies_by_class)
            classwise_test_clean_accuracies_for_all_epochs.append(ema_test_clean_accuracies_by_class)
    
    elif weight_average_type == "fawa":
        if epoch >= 49:
            classwise_test_pgd_accuracies_for_all_epochs.append(fawa_test_pgd_accuracies_by_class)
            classwise_test_fgsm_accuracies_for_all_epochs.append(fawa_test_fgsm_accuracies_by_class)
            classwise_test_clean_accuracies_for_all_epochs.append(fawa_test_clean_accuracies_by_class)
    
    else:
        classwise_test_pgd_accuracies_for_all_epochs.append(test_pgd_accuracies_by_class)
        classwise_test_fgsm_accuracies_for_all_epochs.append(test_fgsm_accuracies_by_class)
        classwise_test_clean_accuracies_for_all_epochs.append(test_clean_accuracies_by_class)
    
logging.info("Training is Done")

if weight_average_type == "ema":
    final_clean_accuracies_by_class_last, final_robust_accuracies_by_class_last = final_evaluation(ema_model, test_loader, mean, std, normalize, dataset_name = dataset_name)
elif weight_average_type == "fawa":
    final_clean_accuracies_by_class_last, final_robust_accuracies_by_class_last = final_evaluation(fawa_model, test_loader, mean, std, normalize, dataset_name = dataset_name)
else:
    final_clean_accuracies_by_class_last, final_robust_accuracies_by_class_last = final_evaluation(model, test_loader, mean, std, normalize, dataset_name = dataset_name)

logging.info("First Autoattack is Done")

final_clean_accuracies_by_class_best, final_robust_accuracies_by_class_best = final_evaluation(best_model, test_loader, mean, std, normalize, dataset_name = dataset_name)

logging.info("Autoattack is Done")
if dataset_name == "cifar10":
    logging.info("Average clean accuracy for best model = {avg_clean_best}".format(avg_clean_best = sum(final_clean_accuracies_by_class_best)/len(final_clean_accuracies_by_class_best)))
    logging.info("Worst clean accuracy for best model = {worst_clean_best}".format(worst_clean_best = min(final_clean_accuracies_by_class_best)))
    logging.info("Average robust accuracy for best model = {avg_robust_best}".format(avg_robust_best = sum(final_robust_accuracies_by_class_best)/len(final_robust_accuracies_by_class_best)))
    logging.info("Worst robust accuracy for best model = {worst_robust_best}".format(worst_robust_best = min(final_robust_accuracies_by_class_best)))
    logging.info("Average clean accuracy for last model = {avg_clean_last}".format(avg_clean_last = sum(final_clean_accuracies_by_class_last)/len(final_clean_accuracies_by_class_last)))
    logging.info("Worst clean accuracy for last model = {worst_clean_last}".format(worst_clean_last = min(final_clean_accuracies_by_class_last)))
    logging.info("Average robust accuracy for last model = {avg_robust_last}".format(avg_robust_last = sum(final_robust_accuracies_by_class_last)/len(final_robust_accuracies_by_class_last)))
    logging.info("Worst robust accuracy for last model = {worst_robust_last}".format(worst_robust_last = min(final_robust_accuracies_by_class_last)))
elif dataset_name == "tiny_imagenet"):
    logging.info("Average clean accuracy for best model = {avg_clean_best}".format(avg_clean_best = sum(final_clean_accuracies_by_class_best)/len(final_clean_accuracies_by_class_best)))
    logging.info("Worst clean accuracy for best model = {worst_clean_best}".format(worst_clean_best = get_average_of_min_20_percent(final_clean_accuracies_by_class_best)))
    logging.info("Average robust accuracy for best model = {avg_robust_best}".format(avg_robust_best = sum(final_robust_accuracies_by_class_best)/len(final_robust_accuracies_by_class_best)))
    logging.info("Worst robust accuracy for best model = {worst_robust_best}".format(worst_robust_best = get_average_of_min_20_percent(final_robust_accuracies_by_class_best)))
    logging.info("Average clean accuracy for last model = {avg_clean_last}".format(avg_clean_last = sum(final_clean_accuracies_by_class_last)/len(final_clean_accuracies_by_class_last)))
    logging.info("Worst clean accuracy for last model = {worst_clean_last}".format(worst_clean_last = get_average_of_min_20_percent(final_clean_accuracies_by_class_last)))
    logging.info("Average robust accuracy for last model = {avg_robust_last}".format(avg_robust_last = sum(final_robust_accuracies_by_class_last)/len(final_robust_accuracies_by_class_last)))
    logging.info("Worst robust accuracy for last model = {worst_robust_last}".format(worst_robust_last = get_average_of_min_20_percent(final_robust_accuracies_by_class_last)))

final_hf = datasets.Dataset.from_pandas(final_df)

final_hf.push_to_hub("Holmeister/{training_type}_{attack_type}_{weight_average_type}_results".format(training_type = training_type,
                        attack_type = attack_type, weight_average_type = weight_average_type), token = hf_token)

if detailed_statistics == "True":
    pgd_df = pd.DataFrame(classwise_test_pgd_accuracies_for_all_epochs)
    fgsm_df = pd.DataFrame(classwise_test_fgsm_accuracies_for_all_epochs)
    clean_df = pd.DataFrame(classwise_test_clean_accuracies_for_all_epochs)
    detailed_df = pd.concat([pgd_df, fgsm_df, clean_df], ignore_index = True)
    detailed_hf = datasets.Dataset.from_pandas(detailed_df)
    detailed_hf.push_to_hub("Holmeister/{training_type}_{attack_type}_{weight_average_type}_detailed_results".format(training_type = training_type,
                            attack_type = attack_type, weight_average_type = weight_average_type), token = hf_token)


logging.info("Push to hub is done")
