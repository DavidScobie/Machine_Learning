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
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.ensemble import RandomForestRegressor

#b

traindata = pd.read_csv('ADNI_CW1_TRAIN.csv')
testdata = pd.read_csv('ADNI_CW1_TEST.csv')

#seperate y and x into separate dataframes
response = traindata['BRAAK34_SUVR']
test_response = testdata['BRAAK34_SUVR']

column_headers = list(traindata)
inputs_list = column_headers[4:]
features = traindata[inputs_list]

#normailse training data
scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,len(response))[0]

#normailse test data
test_features = testdata[inputs_list]
test_scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(test_features[[feature]]).transform(test_features[[feature]])
    test_scale_feat[feature] = scaler.reshape(1,len(testdata['BRAAK34_SUVR']))[0]

train_r2=[]
test_r2=[]

#train the random forest model
regr = RandomForestRegressor(n_estimators=500,min_samples_split=5)
regr.fit(np.array(scale_feat),np.array(response))

#find y_pred for training data
Ypred_train=[]
rowy=[]
for i in range (len(traindata)):
    for j in range (len(inputs_list)):
        rowy.append([scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_train.append(regr.predict([flat_list]))
    rowy=[]
    
traindata['YPred']=Ypred_train

#Find the R2 score for Y_pred
train_r2.append(r2_score(np.array(response),Ypred_train))
print(train_r2)

#find y_pred for test data
Ypred_test=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_test.append(regr.predict([flat_list]))
    rowy=[]
    
testdata['YPred']=Ypred_test

#Find the R2 score for Y_pred_test
test_r2 = r2_score(np.array(test_response),Ypred_test)
print(test_r2)

#Out of bag estimate
regr_oob = RandomForestRegressor(n_estimators=500,min_samples_split=5, oob_score=True)
regr_oob.fit(np.array(scale_feat),np.array(response))
print(regr_oob.oob_score_)


