
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


# In[2]:


X = pd.read_csv(sys.argv[3],low_memory=False)
X_test = pd.read_csv(sys.argv[5],low_memory=False)
Y = np.loadtxt(sys.argv[4],dtype=np.int,delimiter=',')
Y_logistic = np.reshape(Y,(-1,1))
Y = Y.flatten()

X=np.array(X,dtype = float)
X_test = np.array(X_test,dtype = float)


X_mean = np.mean(X,axis =0,keepdims = True)
X_std = np.std(X,axis = 0,keepdims = True)

X = (X-X_mean)/X_std
X_test = (X_test-X_mean)/X_std

print("X_shape:",X.shape,"\nY_shape:",Y.shape)


# In[3]:


model = GradientBoostingClassifier(n_estimators = 600,verbose  =1)
model.fit(X,Y)


# In[4]:


result = model.predict(X_test)


# In[5]:


with open(sys.argv[6], 'w') as f:
       print('id,label', file=f)
       for (i, p) in enumerate(result) :
           print('{},{}'.format(i+1, p), file=f)



