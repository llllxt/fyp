from __future__ import absolute_import
import os
import pandas as pd
import torch
import numpy as np
import torchvision
import datetime
import csv
import torch.nn as nn
import time
import argparse
import random
import copy
import config_task
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,average_precision_score,precision_recall_fscore_support
from torchvision import transforms,models
from models import resnet26
from data import CustomDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,average_precision_score,precision_recall_fscore_support
from utils.evaluations import save_results,do_roc,do_prc,do_prg,get_percentile
from torch.utils.data import DataLoader

def recursion_change_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
        else:
            for name, module1 in module._modules.items():
                module1 = recursion_change_bn(module1)


def train_model(model,critetion,optimizer,scheduler,dataloaders,device,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print('-'*10)

        #each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase== 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #zero the parameter gradients
                optimizer.zero_grad()

                #forward
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #get model outputs and calculate loss
                    outputs = model(inputs)
                    loss= criterion(outputs,labels)

                    _,preds= torch.max(outputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input.size(0)
                running_corrects += torch.sum(preds== labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss,epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
    time_elapsed = time.time - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed%60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #load best model weight
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict,"model.pkl")
    return model,val_acc_history

def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes,feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == "resnet26":
        model = resnet26()        
        print(model)           
        file_path = "resnet26_pretrained.t7"
        checkpoint = torch.load(file_path)
        model = checkpoint['net']
        for name, module in model._modules.items():
            recursion_change_bn(model)
        # print(checkpoint)
        # mapped_state_dict = OrderedDict()
        # for key, value in checkpoint['net']._modules.items():
        #     print(key)
        #     mapped_key = key
        #     mapped_state_dict[mapped_key] = value
        #     if 'running_var' in key:
        #         mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1).to(device)
        # model.load_state_dict(mapped_state_dict)
        
        set_parameter_requires_grad(model,feature_extract)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.linears = nn.ModuleList([nn.Linear(int(256*config_task.factor), 2)])
        
    return model
def train(model_name,num_classes,feature_extract,num_epochs):
    model_ft = initialize_model(model_name,num_classes,feature_extract,use_pretrained=True)
    print(model_ft)

    print("initializing datasets and dataloaders")
    data = {
    'train':CustomDataset(split="train"),
    'test':CustomDataset(split="test")

    }
    dataloaders = {
    'train':DataLoader(data['train'],batch_size=20,shuffle=True),
    'test':DataLoader(data['test'],batch_size=20,shuffle=True)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    optimizer_ft = optim.SGD(params_to_update,lr=0.001,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    #train and evaluate 
    model_ft, hist = train_model(model_ft,dataloaders,criterion,optimizer_ft,dataloaders,device,num_epochs=num_epochs)

if __name__ == "__main__":
    model_name = "resnet26"
    num_classes = 2
    feature_extract = True
    num_epochs = 1

    train(model_name,num_classes,feature_extract,num_epochs)

