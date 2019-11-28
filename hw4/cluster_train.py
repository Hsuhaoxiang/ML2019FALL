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
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(1600, 256)
        
        
        
        # define: decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 5, 2, dilation=2),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, dilation=2),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 6, 1, dilation=1),
            nn.Sigmoid(),
        )


    def encoder(self, x):
        x1 = self.conv1(x)
        #print(x1.shape)
        x2 = self.conv2(x1)
        #print(x2.shape)
        x3 = self.conv3(x2)
        #print(x3.shape)
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
        z =  z.view(-1, 64, 2, 2)
        x4 = self.decoder1(z)
        #print(x4.shape)
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

# detect is gpu available.
use_gpu = torch.cuda.is_available()
if use_gpu:
    device =torch.device('cuda:0')
else:
    device =torch.device("cpu")


# load data and normalize to [-1, 1]
trainX = np.load('./trainX.npy')
print(trainX.shape)
trainX = np.transpose(trainX, (0, 3, 1, 2))/255.
trainX = torch.Tensor(trainX)


# if use_gpu, send model / data to GPU.

# Dataloader: train shuffle = True
train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD



def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
    loss = nn.L1Loss(reduction='sum')
#     MSE = F.mse_loss(recon_x, x, size_average=False)
    l1_loss = loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1_loss + KLD, KLD, l1_loss
# We set criterion : L1 loss (or Mean Absolute Error, MAE)

model = VAE()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# Now, we train 20 epochs.
for epoch in range(1):
    model.train()
    total_loss, best_loss = 0, 100
    """csie ta code
    for x in train_dataloader:

        latent, reconstruct = model(x)
        loss = criterion(reconstruct, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumulate_loss = loss.item() * x.shape[0]

    print(f'Epoch { "%03d" % (epoch+1) }: Loss : { "%.8f" % (cumulate_loss / trainX.shape[0])}')
    """

    for idx, image in enumerate(train_dataloader):
        reconsturct , mean, var = model(image)
        loss, bce, kld = loss_function(reconsturct, image, mean, var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += (loss.item() / len(train_dataloader))
        print('[Epoch %d | %d/%d] loss: %.8f' %((epoch+1), idx*32, len(train_dataloader)*32, loss.item()), end='\r')
    print("\n  Training  | Loss:%.4f \n\n" % total_loss)
    

    # Collect the latents and stdardize it.
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

torch.save(latent_vec,"latent_vec_new2222.npy")
