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
    def __init__(self,seed):
        self.seed = seed
        
        self.dataset="dr_kaggle"
        self.anomaly_type = "novelty"
        self.a = 0


    # def train_model(self,model,critetion,optimizer,scheduler,num_epochs=25):
    #   since = time.time()
    #   best_model_wts = copy.deepcopy(model.state_dict())


    #   for epoch in range(num_epochs):
    #       print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #         print('-' * 10)

    #         for phase in ['train','val']:
    #           if phase == 'train':
    #               model.train()
    #           else:
    #               model.eval()

    #           running_loss = 0.0
    #             running_corrects = 0

    #           for inputs, labels in dataloaders[phase]:
    #               inputs = inputs.to(device)
    #               labels = labels.to(device)

    #               optimizer.zero_grad()

    #               #forward
    #               with torch.set_grad_enabled(phase=="train"):
    #                   outputs = model(inputs)
    #                   _,preds = torch.max(outputs,1)
    #                   loss = critetion(outputs,labels)

    #                   #backward + optimize only if in training phase
    #                   if phase == 'train':
    #                       loss.backward()
    #                       optimizer.step()

    #               # statistics
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)

    #                 if phase == 'train':
    #                   scheduler.step()

    #               epoch_loss = running_loss / dataset_sizes[phase]
    #               epoch_acc = running_corrects.double() / dataset_sizes[phase]

    #               print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #                   phase, epoch_loss, epoch_acc))
    #   time_elapsed = time.time() - since
    #     print('Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60))
    #     print('Best val Acc: {:4f}'.format(best_acc))

    #     # load best model weights
    #     model.load_state_dict(best_model_wts)
    #     return model

    def recursion_change_bn(self,module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
        else:
            for name, module1 in module._modules.items():
                module1 = self.recursion_change_bn(module1)

    def extract_feature(self):
        model = models.resnet50(pretrained=True)
        model.eval()
        print(model)
        # model.cuda()
        # model = FeatureExtractor(model,'avgpool')
        # file_path = "resnet26_pretrained.t7"
        # checkpoint = torch.load(file_path)
        # model = checkpoint['net']
        # for name, module in model._modules.items():
            # self.recursion_change_bn(model)
        # model = torch.nn.Sequential(*(list(model.children())[:-1]))

        model = FeatureExtractor(model,'avgpool').cuda()
        # model.eval()
        #load data
        data = {
        'train':CustomDataset(split="train"),
        'test':CustomDataset(split="test")

        }
        dataloaders = {
        'train':DataLoader(data['train'],batch_size=20,shuffle=True),
        'test':DataLoader(data['test'],batch_size=20,shuffle=False)
        }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        feature_list = []
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
        x_train,x_test = self.extract_feature()
       
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()
        print("##################x_train###############")
        print(x_train.shape)
        print(x_test.shape)
        print(x_train)
        print(x_test)
        model2=IsolationForest(contamination='auto',behaviour='new')
        print("training...")
        model2.fit(x_train)
        print("predicting...")
        y_pred_train = model2.predict(x_train)
        y_pred_test = model2.predict(x_test)
        scores = -model2.decision_function(x_test)
        scorestrain = -model2.decision_function(x_train)
        return scores,y_pred_test

    def evaluation(self):
        a = 0
        scores, y_pred_test = self.isolation_forest()
        print("##########scores###########")
        print(scores)
        self.testy = np.load("test_y.npy")
        print("##############testy##############")
        print(self.testy)
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
    args = parser.parse_args()
    seed= args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Transfer_Forest(seed)
    model.evaluation()


    