import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import glob
import time
import copy



def jpg_image_to_array(image_path,sz):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  img = Image.open(image_path)

  # re-sizing:  
  if img.size[0] != sz:
      img = img.resize(size = (sz,sz) )
    
  # convert to RGB if necessary
  if img.mode != 'RGB':
        img = img.convert(mode='RGB')
  # convert img in array type
  im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
  #print(im_arr.shape)
  im_arr = im_arr.reshape((img.size[1],img.size[0],-1))
  #assert im_arr.shape == (sz,sz,3)

  return im_arr 


def flat_to_img(x,w,h,c):
    img = x.reshape(x.shape[0],w,h,c)
    
def flatten(img):
    N = img.shape[0]
    img.reshape(N,-1)
    
def range_pixel(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 0:
                img[i][j] = 0
            if img[i][j] > 255:
                img[i][j] = 255
                
                
def change_range(X,NewMin,NewMax):
    OldMin = np.amin(X)
    OldMax = np.amax(X)
    OldRange = OldMax - OldMin
    NewRange = NewMax - NewMin
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            NewValue = (((X[i,j] - OldMin) * NewRange) / OldRange) + NewMin
            X[i,j] = NewValue

            
def delete_out_of_range(X,low,up):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j]<low:
                X[i,j] = low
            if X[i,j] > up:
                X[i,j] = up   
                
                
def resize(set_img, sz, dataset):
    new_set_img = []
    for i in range(set_img.shape[0]):
        x = cv2.resize(set_img[i], dsize=(sz,sz), interpolation=cv2.INTER_CUBIC)
        if dataset=='mnist':
            x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
        new_set_img.append(x)
    return np.array(new_set_img)

    
    
    
    
    
    