import os
import glob
import pickle
import random

###############sample train data for finetune################
split = 'train'
root_dir = "/home/students/student3_15/00_astar/00_baseline/00_drkaggle"
print("sampling data...")
imgs,lbls = [], []
folder_name = 'prepBG_'+split
img_path = os.path.join(root_dir,folder_name)
files_name = glob.glob(os.path.join(img_path,"*.jpeg"))
files_train = random.sample(files_name,10000)
with open('finetune.txt', 'w') as f:
    for item in files_train:
        print >> f, item

#############sample train data for isolation##############
files_test = random.sample(files_name,7000)
file_iso = []
for file in files_test:
	if file not in files_train:
		file_iso.append(file)
print(len(file_iso))
with open('isolation.txt', 'w') as f:
    for item in files_train:
        print >> f, item

############sample test data for isolation
split='test'
root_dir = "/home/students/student3_15/00_astar/00_baseline/00_drkaggle"
print("sampling data...")
imgs,lbls = [], []
folder_name = 'prepBG_'+split
img_path = os.path.join(root_dir,folder_name)
files_name = glob.glob(os.path.join(img_path,"*.jpeg"))
files_train = random.sample(files_name,10000)
print(len(files_train))
with open('test.txt', 'w') as f:
    for item in files_train:
        print >> f, item
