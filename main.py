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
from sklearn.metrics import classification_report, confusion_matrix

def recursion_change_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
        else:
            for name, module1 in module._modules.items():
                module1 = recursion_change_bn(module1)


def train_model(model,optimizer,dataloaders,device,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print("before training ..")
    for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    i=0
    for epoch in range(num_epochs):
        
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print('-'*10)

        

        #each epoch has a training and validation phase
        for phase in ['train']:
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
                    outputs = model(inputs)                  
                    criterion = nn.CrossEntropyLoss()
                    loss= criterion(outputs,labels)
                    _,preds= torch.max(outputs,1)
                   
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds== labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss,epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                
                best_acc = epoch_acc
                print("saving best weight ...")
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,"model_linear_balance.pkl")

            
    print('Best val Acc: {:4f}'.format(best_acc))
    #load best model weight
    model.load_state_dict(best_model_wts)
    return model
def test_model(model):
    model.eval()
    model.cuda()
    data = {
    'train':CustomDataset(split="train",seed=0,step='finetune'),
    'test':CustomDataset(split="test",seed=0,step='finetune')

    }
    dataloaders = {
    'train':DataLoader(data['train'],batch_size=20,shuffle=True),
    'test':DataLoader(data['test'],batch_size=20,shuffle=False)
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testy=[]
    y_pred=[]
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #zero the parameter gradients

        #forward
        #track history if only in train
        total = 0
        correct = 0
        with torch.set_grad_enabled(False):
            outputs = model(inputs)                  
            _,preds= torch.max(outputs,1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            testy+=labels.tolist()
            y_pred+=preds.tolist()
    print(testy)
    print("*************************8")
    print(y_pred)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    cm = confusion_matrix(testy, y_pred, labels=None, sample_weight=None)
    print('confusion matrix:',cm)

def set_parameter_requires_grad(model,feature_extracting):
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes,feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0
    net = resnet26()
    if model_name == "resnet26":
        checkpoint = torch.load("resnet26_pretrained.t7")
        net_old = checkpoint['net']

        store_data = []
        t = 0
        for name, m in net_old.named_modules():
            if isinstance(m, nn.Conv2d):
                store_data.append(m.weight.data)
                t += 1

        element = 0
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                element += 1

        element = 1
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                element += 1

        store_data = []
        store_data_bias = []
        store_data_rm = []
        store_data_rv = []
        for name, m in net_old.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                store_data.append(m.weight.data)
                store_data_bias.append(m.bias.data)
                store_data_rm.append(m.running_mean)
                store_data_rv.append(m.running_var)

        element = 0
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
                    m.weight.data = torch.nn.Parameter(store_data[element].clone())
                    m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                    m.running_var = store_data_rv[element].clone()
                    m.running_mean = store_data_rm[element].clone()
                    element += 1

        element = 1
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
                    m.weight.data = torch.nn.Parameter(store_data[element].clone())
                    m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                    m.running_var = store_data_rv[element].clone()
                    m.running_mean = store_data_rm[element].clone()
                    element += 1
        
       

    # if model_name == "resnet26":
    #     model = resnet26()        
    #     print(model)           
    #     file_path = "resnet26_pretrained.t7"
    #     checkpoint = torch.load(file_path)
    #     model = checkpoint['net']
    #     for name, module in model._modules.items():
    #         recursion_change_bn(model)
        # print(checkpoint)
        # mapped_state_dict = OrderedDict()
        # for key, value in checkpoint['net']._modules.items():
        #     print(key)
        #     mapped_key = key
        #     mapped_state_dict[mapped_key] = value
        #     if 'running_var' in key:
        #         mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1).to(device)
        # model.load_state_dict(mapped_state_dict)
        
        set_parameter_requires_grad(net,feature_extract)
        # torch.save(net.state_dict(),"model_notrain.pkl")
        # factor = config_task.factor
        # net.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256*factor)),nn.ReLU(True)) for i in range(1)])
        #net.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256)),nn.ReLU(True))])
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        net.linears = nn.ModuleList([nn.Linear(256, 2)])

        
    return net
def train(model_name,num_classes,feature_extract,num_epochs,batchsize):
    model_ft = initialize_model(model_name,num_classes,feature_extract,use_pretrained=True)
    print(model_ft)

    print("initializing datasets and dataloaders")
    data = {
    'train':CustomDataset(split="train",seed=0,step='finetune'),
    'test':CustomDataset(split="test",seed=0,step='finetune')

    }
    dataloaders = {
    'train':DataLoader(data['train'],batch_size=batchsize,shuffle=True),
    'val':DataLoader(data['test'],batch_size=batchsize,shuffle=True)
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
    
    #train and evaluate 
    model_ft = train_model(model_ft,optimizer_ft,dataloaders,device,num_epochs=num_epochs)
    return model_ft

if __name__ == "__main__":
    model_name = "resnet26"
    num_classes = 4
    feature_extract = True
    num_epochs = 30
    batchsize =20

    best_model = train(model_name,num_classes,feature_extract,num_epochs,batchsize)
    torch.save(best_model.state_dict(),"model_linear_balance.pkl")
    # best_model = resnet26()
    # checkpoint = torch.load("model_linear.pkl")
    # best_model.load_state_dict(checkpoint)
    
    test_model(best_model)

