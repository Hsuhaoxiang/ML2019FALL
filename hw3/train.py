
# coding: utf-8


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


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


# In[3]:


class FaceImage(Dataset):
    def __init__(self,label,picture_file,path,Train=True):
        self.data_path = path
        self.label = label
        self.train = Train
        self.picture_file = picture_file
    def __getitem__(self,index):

        file_path = os.path.join(self.data_path,self.picture_file[index])
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img,0)
        img =torch.tensor(img).float()
        if self.train==True:
            return torch.FloatTensor(img),self.label[index,1]
        else:
            return img

    def __len__(self):
        return len(os.listdir(self.data_path))


# In[4]:


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
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride =(1,1)),
            nn.Dropout2d(0.4),
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
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        out = self.fcn3(x)
        
        return out


# In[5]:


train_label = pd.read_csv(sys.argv[2])
train_label = np.array(train_label,dtype = int)
datalen  = len(train_label)
picture_file = sorted(os.listdir(sys.argv[1]))
train_dataset = FaceImage(train_label,picture_file,sys.argv[1])
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 256)


# In[7]:


model = ImageNet().to(device)
#model = torch.load("Imagenet3.pkl")
print(model)
with open("loss_acc.csv","w") as f:
    for epoch in range(800):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        train_loss = 0
        correct = 0
        train_loss = []
        train_acc = []
        for batch_index ,(data, label) in enumerate(train_loader):
            data, label =data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_loss: {:.6f},train_acc：{:.6f}'.format(epoch, batch_index * len(data), len(train_loader.dataset),100. * batch_index / len(train_loader),np.mean(train_loss), np.mean(train_acc)),end = "\r")

        print("train_loss: {:.6f},train_acc：{:.6f}".format(np.mean(train_loss), np.mean(train_acc)) ,file = f)


torch.save(model,"result.pkl")




