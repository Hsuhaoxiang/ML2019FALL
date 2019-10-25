#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import sys
import math

X = pd.read_csv("X_train",low_memory=False)
X_test = pd.read_csv("X_test",low_memory=False)
Y = np.loadtxt("Y_train",dtype=np.float,delimiter=',')


Y = np.reshape(Y,(-1,1))
X=np.array(X,dtype = float)
X_test = np.array(X_test,dtype = float)

#normalize
"""
X_mean = np.mean(X,axis =0,keepdims = True)
X_std = np.std(X,axis = 0,keepdims = True)
X = (X-X_mean)/X_std
X_test = (X_test-X_mean)/X_std
print(X,X_test)
"""
#Y = pd.read_csv("Y_train",low_memory=False)
#Y = np.array(Y,dtype = float)

print("X_shape:",X.shape,"\nY_shape:",Y.shape)


# In[56]:


class Generative_model():
    def __init__(self):
        pass
    
    
    def sigmoid(self,z):#   sigmod(z) = 1/(1+e^-z)
        return 1/(1+np.exp(-z))
    

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]        
        x1 = []#Y=1
        x2 = []#Y=0
        for i in range(X.shape[0]):
            if Y[i]==1:
                x1.append(X[i,:])
            else:
                x2.append(X[i,:])
        x1 = np.array(x1,dtype = float)#~7000*106
        x2 = np.array(x2,dtype = float)#~20000*106
        
        
        self.X_mean_class1 = np.mean(x1,axis =0,keepdims = True)#1*106
        self.X_mean_class2 = np.mean(x2,axis =0,keepdims = True)#1*106

        self.X_len = x1.shape[0]+x2.shape[0]
        self.Pc1 = (x1.shape[0]/X.shape[0])
        self.Pc2 = (x2.shape[0]/X.shape[0])
        
        Sigma1 = np.dot((x1 - self.X_mean_class1).T,(x1 - self.X_mean_class1))/x1.shape[0]
        Sigma2 = np.dot((x2 - self.X_mean_class2).T,(x2 - self.X_mean_class2))/x2.shape[0]
        self.Sigma = self.Pc1*Sigma1 + self.Pc2*Sigma2
        #print("Sigma1:",Sigma1.shape,"\nSigma2:",Sigma2.shape,"\nSigma:",self.Sigma.shape)

    def predict(self, test_X):
        #print("testx:",test_X.shape)
        sigma_inverse = np.linalg.pinv(self.Sigma)
        #print(sigma_inverse)
        w = np.dot((self.X_mean_class1-self.X_mean_class2),sigma_inverse)
        #print("w=",w)
        b = (-0.5) * np.dot(np.dot(self.X_mean_class1, sigma_inverse), self.X_mean_class1.T) + (0.5) * np.dot(np.dot(self.X_mean_class2, sigma_inverse), self.X_mean_class2.T) + np.log(float(self.Pc1)/(self.Pc2))
        z = np.dot(w, X_test.T) + b        
        
        return self.sigmoid(z)


# In[57]:


model = Generative_model()
model.fit(X,Y)
result = model.predict(X_test)
result = np.reshape(result,-1)

for i in range(len(result)):
    if result[i]<0.5:
        result[i] = 0
    else:
        result[i] = 1
result =np.array(result,dtype=int)



# In[58]:


with open('generative_model.csv', 'w') as f:
       print('id,label', file=f)
       for (i, p) in enumerate(result) :
           print('{},{}'.format(i+1, p), file=f)


# # WHY inv can't work?

# In[ ]:




