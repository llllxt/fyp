from __future__ import absolute_import
from __future__ import absolute_import
import random
import pandas as pd
import numpy as np
from PIL import Image

from pandas import Series, DataFrame
from torch.utils.data import Dataset
from torchvision import transforms, utils
import os
import glob

# normalize= transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )
preprocess = transforms.Compose([
    transforms.ToTensor(),
    
    ])
def get_oneclass_dataset(split,step,files_name,labels):
    imgs,lbls = [], []
    for name in files_name:
        full_name=name.split('/')[-1]
        img_name = full_name.split('.')[0]
        if labels[labels['image']==img_name]['level'].values[0] == 1:
            continue
        elif labels[labels['image']==img_name]['level'].values[0] == 0:
            lbls.append(0)
            imgs.append(name)

        elif labels[labels['image']==img_name]['level'].values[0] == 2:
            continue
        elif labels[labels['image']==img_name]['level'].values[0] == 3:
            continue
        elif labels[labels['image']==img_name]['level'].values[0] == 4:
            if step == 'finetune':
                continue
            else:
                lbls.append(1)
                imgs.append(name)
        else:
            raise Exception("PROBLEM"+name)
            continue
        numpy_labels = np.array(lbls)
        np.save(split+"_x.npy",imgs)
        np.save(split+"_y.npy",lbls)
    print(set(lbls))
    return imgs,lbls
def get_balance_data(split,step,files_name,labels):
    imgs,lbls = [], []
    count = 0
    for name in files_name:
        full_name=name.split('/')[-1]
        img_name = full_name.split('.')[0]
        if labels[labels['image']==img_name]['level'].values[0] == 1:
            continue
        elif labels[labels['image']==img_name]['level'].values[0] == 0:
            count += 1
            if(step=="finetune" and split=="train" and count > 900):
                continue
            else:
                lbls.append(0)
                imgs.append(name)
            
        elif labels[labels['image']==img_name]['level'].values[0] == 2:
            continue
        elif labels[labels['image']==img_name]['level'].values[0] == 3:
            if step != 'finetune':
                continue
            else:
                lbls.append(1)
                imgs.append(name)
       
        elif labels[labels['image']==img_name]['level'].values[0] == 4:
            lbls.append(1)
            imgs.append(name)
        else:
            raise Exception("PROBLEM"+name)
            continue
        numpy_labels = np.array(lbls)
        np.save(split+"_x.npy",imgs)
        np.save(split+"_y.npy",lbls)
    print(set(lbls))
    return imgs,lbls

def get_dataset(split,seed,step):
    root_dir = "/home/students/student3_15/00_astar/00_baseline/00_drkaggle"
    print("sampling data...")
    folder_name = 'prepBG_'+split
    img_path = os.path.join(root_dir,folder_name)
    files_name = glob.glob(os.path.join(img_path,"*.jpeg"))
    #load image paths
    if split == "train":
        files_name = []
        file = step+".txt"
        with open(file, "r") as f:
            for line in f:
                files_name.append(line.strip())
    else:
        files_name = []
        with open('test.txt', "r") as f:
            for line in f:
                files_name.append(line.strip())
    labels = pd.read_csv(os.path.join(root_dir,split+'Labels.csv'),delimiter=',')
    imgs, lbls = get_balance_data(split,step,files_name,labels)
    return imgs,lbls



def default_loader(path):   
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor
class CustomDataset(Dataset):
    def __init__(self,split,seed,step,loader=default_loader):
        self.loader=loader
        self.split = split
        self.seed  = seed
        self.images,self.labels = get_dataset(split,seed,step)

    def __getitem__(self,index):
        fn=self.images[index]
        img = self.loader(fn)
        label = self.labels[index]
        return img,label
    def __len__(self):
        return len(self.images)


