#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import pandas as pd
import sys
import cv2
import sys


# In[2]:


class FaceImage(Dataset):
    def __init__(self,picture_file,path):
        self.data_path = path
        self.picture_file = picture_file
    def __getitem__(self,index):

        file_path = os.path.join(self.data_path,self.picture_file[index])
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img,0)
        img =torch.tensor(img).float()
        return img

    def __len__(self):
        return len(os.listdir(self.data_path))


# In[3]:


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3,stride =(1,1)),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3,stride =(1,1)),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride =(1,1)),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fcn1 = nn.Linear(in_features =4096 ,out_features =512 ,bias = True)
        self.fcn2 = nn.Linear(in_features = 512,out_features =256 ,bias = True)
        self.fcn3 = nn.Linear(in_features =256 ,out_features =7 ,bias = True)

    def forward(self , x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        #print("size:",x.shape)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        out = self.fcn3(x)
        
        return out


# In[4]:


picture_file = sorted(os.listdir(sys.argv[1]))
test_dataset = FaceImage(picture_file,sys.argv[1])

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size =256)


# In[8]:


model = torch.load("Imagenet_best.pkl",map_location = "cpu")
model = model.cpu()


# In[9]:


model.eval()
ans = []
for idx,test in enumerate(test_loader):
    test =test
    ans.append(model(test))


# In[1]:


ans_final =[]
for a in ans:
    for pre in a:
        pre = pre.tolist()
        ans_final.append(pre.index(max(pre)))


# In[11]:


with open(sys.argv[2],"w") as f:
    print("id,label", file = f)
    for id,label in enumerate(ans_final):
        print("{},{}".format(id,label) ,file = f)





