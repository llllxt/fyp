from __future__ import absolute_import
import os
import pandas as pd
import torch
import numpy as np
import torchvision
import datetime
import csv
import torch.nn as nn

import argparse
import random
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

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)  # last layer output put into current layer input
            if name == self.extracted_layers:
                return x
class Transfer_Forest():
    def __init__(self,seed,model_name,gpu):
        self.seed = seed
        self.dataset="dr_kaggle"
        self.anomaly_type = "novelty"
        self.a = 0
        self.model_name = model_name
        self.gpu = gpu

    def recursion_change_bn(self,module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
        else:
            for name, module1 in module._modules.items():
                module1 = self.recursion_change_bn(module1)

    def extract_feature(self,model_namem):
        model = resnet26()       
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint)
        model.cuda()
        model.eval()
        #load data
        data = {
        'train':CustomDataset(split="train",seed=42,step='isolation'),
        'test':CustomDataset(split="test",seed=42,step='isolation')

        }
        dataloaders = {
        'train':DataLoader(data['train'],batch_size=20,shuffle=True),
        'test':DataLoader(data['test'],batch_size=20,shuffle=False)
        }

        device = torch.device("cuda:"+self.gpu if torch.cuda.is_available() else "cpu")
        
        feature_list = []
        train= []
        test = []
        with torch.no_grad():
            for split in ['train','test']:           
                for i, (inputs,labels) in enumerate(dataloaders[split]):
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # x_feature_batchs= model(inputs)
                    x_feature_batchs= model(inputs)
                   
                    if i:
                        features = np.concatenate([features,x_feature_batchs],axis=0)
                    else:
                        features = x_feature_batchs
                feature_list.append(features)

        np.save("feature1.npy",feature_list[0])
        np.save("feature2.npy",feature_list[1])
        return feature_list[0],feature_list[1]

    def isolation_forest(self):
        x_train,x_test = self.extract_feature(self.model_name)
       
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()
        print("##################x_train###############")
        print(x_train.shape)
        print(x_test.shape)
        # print(x_train)
        # print(x_test)
        
        model2=IsolationForest(contamination='auto',behaviour='new')
        print("training...")
        model2.fit(x_train)
        print("predicting...")
        # y_pred_train = model2.predict(x_train)
        y_pred_test = model2.predict(x_test)
        scores = -model2.decision_function(x_test)
        # scorestrain = -model2.decision_function(x_train)
        return scores,y_pred_test

    def evaluation(self):
        a = 0
        scores, y_pred_test = self.isolation_forest()
        # print("##########scores###########")
        # print(scores)
        self.testy = np.load("test_y.npy")
        # print("##############testy##############")
        # print(self.testy)
        auroc = do_roc(scores=scores, true_labels=self.testy, file_name='roc', directory='data', plot=False)
        print('AUROC:', auroc)
        auprc = do_prc(scores, self.testy, file_name='auprc',
                directory='data')
        auprg = do_prg(scores, self.testy, file_name='auprg',
                directory='data')
        per = get_percentile(scores, self.dataset, self.anomaly_type, a)
        print(per)
        y_pred = (scores>=per).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(self.testy.astype(int),
                                                       y_pred.astype(int),
                                                       average='binary')
        cm = confusion_matrix(self.testy, y_pred, labels=None, sample_weight=None)
        print('confusion matrix:',cm)
        # check  if folder exist:
        if not os.path.exists('results/'):
            os.makedirs('results/')
        if not os.path.exists('results/tf_results'):
            os.makedirs('results/tf_results')
        # write results on csv file
        csv_name = 'results/tf_results/results_'+self.dataset+'_'+self.anomaly_type+'_'+'.csv'
        exists = os.path.isfile(csv_name)
        if not exists:
            columns = ['auroc','auprc','auprg','precision','recall','f1','date']
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_name)
        new_data = [auroc ,auprc, auprg, precision, recall, f1, datetime.datetime.now(),]
        with open(csv_name,'a') as csv_file:
            filewriter =  csv.writer(csv_file)
            filewriter.writerow(new_data)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='random_seed')
    parser.add_argument('--model', default='model_notrain.pkl',required=True,help='model name')
    parser.add_argument('--gpu', default='0',required=True,help='gpu index')
    args = parser.parse_args()
    seed= args.seed
    model_name =args.model
    gpu = args.gpu
    model = Transfer_Forest(seed,model_name,gpu)
    model.evaluation()