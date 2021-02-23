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
from sklearn.svm import SVR

#b

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

#train a linear kernel SVR
lin_regr = SVR(kernel='linear')
lin_regr.fit(np.array(scale_feat),np.array(response))

Ypred_lin_train=[]
rowy=[]
for i in range (len(traindata)):
    for j in range (len(inputs_list)):
        rowy.append([scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_lin_train.append(lin_regr.predict([flat_list]))
    rowy=[]

traindata['YPred_lin']=Ypred_lin_train
# print(traindata['YPred_lin'])

#train a polynomial kernel SVR
poly_regr = SVR(kernel='poly',degree=3)
poly_regr.fit(np.array(scale_feat),np.array(response))

Ypred_poly_train=[]
rowy=[]
for i in range (len(traindata)):
    for j in range (len(inputs_list)):
        rowy.append([scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_poly_train.append(poly_regr.predict([flat_list]))
    rowy=[]

traindata['YPred_poly']=Ypred_poly_train
# print(traindata['YPred_poly'])

#train an rbf kernel SVR
rbf_regr = SVR(kernel='rbf')
rbf_regr.fit(np.array(scale_feat),np.array(response))

Ypred_rbf_train=[]
rowy=[]
for i in range (len(traindata)):
    for j in range (len(inputs_list)):
        rowy.append([scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_rbf_train.append(rbf_regr.predict([flat_list]))
    rowy=[]

traindata['YPred_rbf']=Ypred_rbf_train
print(traindata['YPred_rbf'])