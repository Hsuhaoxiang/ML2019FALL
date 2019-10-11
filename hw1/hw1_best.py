# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sys
from sklearn import ensemble as em
from sklearn.externals import joblib

X_test = pd.read_csv(sys.argv[1])
def preprocess(train):
    train.replace("#",'',inplace=True,regex=True)
    train.replace("x",'',inplace=True,regex=True)
    train.replace("\*",'',inplace=True,regex=True)
    train.replace("NR",0,inplace=True) #將NR變成0
    train.fillna(0, inplace=True) #把NAN變成0
    train = train.iloc[:,2:]
    return train

def get_feature(train,test=False):
    train = np.array(train,dtype=float)# to array 6588*24
    tmp=[]
    feature = []
    for i  in range(train.shape[0]): #0~6587
        index = i%18
        tmp.append(train[i])
        if index==17:
            tmp = np.array(tmp,dtype=float)#shape7*24
            feature.append(tmp)
            tmp = []
    feature = np.array(feature,dtype=float)#366*168
    print(feature.shape)
    return feature

X_test = preprocess(X_test)
X_test = get_feature(X_test)#500*7*9


tmp = []
for i in range(X_test.shape[0]):
    tmp.append(X_test[i].flatten())
X_test = tmp 
X_test = np.array(X_test,dtype=float)#shape 500* 162(9*18)

print(sys.argv[3])
gbmodel = joblib.load(sys.argv[3])


ans = gbmodel.predict(X_test)

for i in range(len(ans)):
    if ans[i]<0:
        ans[i] = 10.445

with open(sys.argv[2], 'w') as f:
	print('id,value', file=f)
	for (i, p) in enumerate(ans):
		print('id_{},{}'.format(i, p),file=f)


