import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNetCV

#a

traindata = pd.read_csv('ADNI_CW1_TRAIN.csv')
testdata = pd.read_csv('ADNI_CW1_TEST.csv')

#seperate y and x into separate dataframes
response = traindata['BRAAK34_SUVR']

column_headers = list(traindata)
inputs_list = column_headers[4:]
features = traindata[inputs_list]

#normailse training data
scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,len(response))[0]

#elastic net
regr = ElasticNet(random_state=0,alpha=0,l1_ratio=0.5)
regr.fit(np.array(scale_feat),np.array(response))
print(regr.coef_)

#test data
test_features = testdata[inputs_list]

#normailse test data
test_scale_feat = pd.DataFrame()
for feature in inputs_list:
    test_scaler = StandardScaler()
    test_scaler = test_scaler.fit(test_features[[feature]]).transform(test_features[[feature]])
    test_scale_feat[feature] = test_scaler.reshape(1,len(test_features))[0]


#Apply test data to the trained linear regression model

rowy=[]
for j in range (len(inputs_list)):
    rowy.append([test_scale_feat[inputs_list[j]][6]])
flat_list = [item for sublist in rowy for item in sublist]
print(regr.predict([flat_list]))

Ypred=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred.append(regr.predict([flat_list]))
    rowy=[]
print(Ypred)
    
testdata['YPred']=Ypred
print(testdata['YPred'])


# regrCV = ElasticNetCV(cv=10,l1_ratio=0.5)
# regrCV.fit(np.array(scale_feat),np.array(response))
# print(regrCV.alpha_)





