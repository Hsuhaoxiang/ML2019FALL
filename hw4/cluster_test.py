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



latent_vec = torch.load("latent_vec_new.npy")

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