import torch
import json
from models import resnet26
import cPickle as pickle
import random
import pandas as pd
import numpy as np
from PIL import Image

from pandas import Series, DataFrame
from torch.utils.data import Dataset
from torchvision import transforms, utils
import os
import glob

normalize= transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
preprocess = transforms.Compose([
    transforms.ToTensor(),
    
    ])
path = ['/home/students/student3_15/00_astar/00_baseline/00_drkaggle/prepBG_train/44214_left.jpeg']
img_pil = Image.open(path[0])
img_pil = img_pil.resize((224,224))
img_tensor = preprocess(img_pil).resize(1,3,224,224)
model = resnet26()
file_path = "model_linear.pkl"
checkpoint = torch.load(file_path)
model.load_state_dict(checkpoint)
print(model(img_tensor))


model1=resnet26()
file_path = "model_notrain.pkl"
checkpoint1= torch.load(file_path)
model1.load_state_dict(checkpoint1)
print(model1(img_tensor))
