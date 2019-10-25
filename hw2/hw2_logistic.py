
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
X = pd.read_csv(sys.argv[3],low_memory=False)

X_test = pd.read_csv(sys.argv[5],low_memory=False)
X=np.array(X,dtype = float)
X_test = np.array(X_test,dtype = float)


X_mean = np.mean(X,axis =0,keepdims = True)
X_std = np.std(X,axis = 0,keepdims = True)

X = (X-X_mean)/X_std
X_test = (X_test-X_mean)/X_std



#Y = pd.read_csv("Y_train",low_memory=False)
#Y = np.array(Y,dtype = float)
Y = np.loadtxt(sys.argv[4],dtype=np.float,delimiter=',')

Y = np.reshape(Y,(-1,1))

bias = np.ones((32561,1))
biastest = np.ones((16281,1))
X = np.concatenate((X,bias),axis = 1)
X_test = np.concatenate((X_test,biastest),axis = 1)
print("X_shape:",X.shape,"\nY_shape:",Y.shape)


# In[2]:


class logistic_regression():
    def __init__(self):
        pass
    def _init_para(self):
        self.W =np.ones((107,1))
        
    def fit(self, X, Y, valid=None, max_epoch=2000, lr=0.000001):
        assert X.shape[0] == Y.shape[0]
        self._init_para()

        for epoch in range(1, max_epoch+1):
            
            W_grad = -np.dot(X.T,(Y-self.predict(X)))
            #print(W_grad.shape)
            self.W = self.W -(lr*W_grad)
            #print(self.W)

            if epoch%100==0:
                training_loss = self.loss(X,Y)
                #accuracy = self.evaluate(X,Y)
                print('[Epoch%5d] - training loss: %5f'%(epoch, training_loss))
            
    def sigmod(self,z):#   sigmod(z) = 1/(1+e^-z)
        return 1/(1+np.exp(-z))
    
    def predict(self, X, test=False):
        return self.sigmod(np.dot(X,self.W))

    def loss(self, X, Y, pred=None):
        predict_value = self.predict(X)
        return -np.sum(Y*np.log(predict_value+0.00000000001)+(1-Y)*(np.log(1-predict_value+0.00000000001)))


    def evaluate(self, X, Y):
        pred=self.predict(X)
        for i in range(len(pred)):
            if pred[i]>=0.5:
                pred[i] = 1
            else:
                pred[i] =0
            return np.mean(1-np.abs(pred-Y))


# In[3]:


model = logistic_regression()
model.fit(X,Y,max_epoch = 10000)


# In[4]:


predict = model.predict(X_test)


print(predict.shape)
for i in range(len(predict)):
    if predict[i]<0.5:
        predict[i] = 0
    else:
        predict[i] = 1
predict =np.array(predict,dtype=int)


# In[5]:


with open(sys.argv[6], 'w') as f:
       print('id,label', file=f)
       for (i, p) in enumerate(predict) :
           print('{},{}'.format(i+1, p[0]), file=f)


# In[ ]:





# In[ ]:




