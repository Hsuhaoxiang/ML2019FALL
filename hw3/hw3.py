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


class FaceImage(Dataset):
    def __init__(self,data_path,label):
        self.data_path = data_path
        self.label = label
    
    def __getitem__(self,index):
        picture_file = '{:0>5d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.paht.join(self.data_der,pic_file),cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img),self.label[index][1]
    def __len__(self):
        return self.label.shape[0]


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size =(5,5),stride =(1,1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size =(4,4),stride =(1,1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size =(3,3),stride =(1,1))
        self.fcn1 = nn.Linear(in_features =256 ,out_features =256 ,bias = True)
        self.fcn2 = nn.Linear(in_features = 256,out_features =256 ,bias = True)
        self.fcn3 = nn.Linear(in_features =256 ,out_features =7 ,bias = True)

    def forwoard(self , x):
        x =F.relu(F.max_pool2d(self.conv1(x),2))
        print('Tensor size and type after conv1:', x.shape, x.dtype)
        x =F.relu(F.max_pool2d(self.conv2(x),2))
        x =F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
train_label = pd.read_csv('./train.csv')
train_dataset = FaceImage(sys.argv[1] ,train_label)
train_loder = torch.utils.data.DataLoader(train_dataset,batch_size = 256)

model = ImageNet()
for epoch in range(5000):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    train_loss = 0
    correct = 0
    for batch_index ,(data, label) in enumerate(train_loder):
        data, label = data.cuda() ,label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        train_loss += F.cross_entropy(output, label).item
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'.format(ep, batch_idx * len(data), len(trainset_loader.dataset),100. * batch_idx / len(trainset_loader), loss.item()))

