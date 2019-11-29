import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import argparse
import csv
import time
import sys
import os
# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data 
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
#from cluster_train import VAE
from sklearn.externals import joblib

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()

        # define: encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        
        # generate mean var
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(1600, 512)
        
        
        
        # define: decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, dilation=2),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, dilation=1),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 5, 1, dilation=1),
            nn.Sigmoid(),
        )


    def encoder(self, x):
        x1 = self.conv1(x)
        #print(x1.shape)
        x2 = self.conv2(x1)
        #print(x2.shape)
        x3 = self.conv3(x2)
        #print("x3",x3.shape)
        return x3
    
    def bottleneck(self,latent):
        latent = latent.view(len(latent), -1)
        mean = self.fc1(latent)
        var = self.fc2(latent)
        z = self.reparameterize(mean,var)
        #print(z.shape)
        return z, mean, var
    
    
    
    def reparameterize(self, mean, var):
        std = var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mean.size())
        z = mean + std * esp
        return z
    
    

    def decoder(self, z):
        #print(z.shape)
        z =  z.view(-1, 32, 4, 4)
        x4 = self.decoder1(z)
        #print("x4",x4.shape)
        x5 = self.decoder2(x4)
        #print(x5.shape)
        x6 = self.decoder3(x5)
        #print(x6.shape)
        return x6
    
    def forward(self, x):
        latent = self.encoder(x)
        z , mean, var = self.bottleneck(latent) 
        rec_ = self.decoder(z)
        return rec_, mean, var

model  = torch.load( 'vae_11171012.pth')


trainX = np.load(sys.argv[1])
print(trainX.shape)
trainX = np.transpose(trainX, (0, 3, 1, 2))/255.
trainX = torch.Tensor(trainX)

test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)

latents = []
latent_sapce = []
for x in test_dataloader:
    _,mu,var = model(x)
    mu = mu.detach().cpu().numpy()
    for i in range(mu.shape[0]):
        latent_sapce.append(mu[i])
        
print('latent_space finish')
latent_space = np.asarray(latent_sapce)

print(latent_space.shape)
latents = (latent_space - np.mean(latent_space, axis=0)) / np.std(latent_space, axis=0)

# Use PCA to lower dim of latents and use K-means to clustering.
pca = PCA(n_components=32, copy=False, whiten=True, svd_solver='full')
latent_vec = pca.fit_transform(latents)

latent_vec = TSNE(n_components = 3,verbose = 5).fit_transform(latent_vec)


print(latent_vec.shape)
#result = SpectralClustering(n_clusters=2,random_state = 2, n_init=10, gamma=1.0).fit(latent_vec)
#result = SpectralClustering(n_clusters=2,random_state = 2, n_init=10, gamma=1.0).fit_predict(latent_vec).labels_
result = KMeans(n_clusters=2, random_state=2, max_iter=1000).fit(latent_vec).labels_


'''
latents = PCA(n_components=16).fit_transform(latents)
result = KMeans(n_clusters = 2).fit(latents).labels_
'''
# We know first 5 labels are zeros, it's a mechanism to check are your answers
# need to be flipped or not.
print(np.sum(result[:5]))
if np.sum(result[:5]) >= 3:
    result = 1 - result
    
""""
if np.sum(result[:5]) != 0 or np.sum(result[:5])!=5:
    print("redo")
"""
# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(sys.argv[2],index=False)
