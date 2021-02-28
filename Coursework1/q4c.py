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

column_headers = list(traindata)
inputs_list = column_headers[4:]
features = traindata[inputs_list]

#normailse training data
scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,len(response))[0]



train_r2=[]
test_r2=[]
alphas = np.linspace(0,0.1,num=11)
for alph in alphas:
    regr = RandomForestRegressor(n_estimators=500,min_samples_split=5)
    regr.fit(np.array(scale_feat),np.array(response))
    # print(regr.coef_)

    #plot R2 against alpha for training
    Ypred_train=[]
    rowy=[]
    for i in range (len(traindata)):
        for j in range (len(inputs_list)):
            rowy.append([scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_train.append(regr.predict([flat_list]))
        rowy=[]
        
    traindata['YPred']=Ypred_train
    # print(traindata['YPred'])

    train_r2.append(r2_score(np.array(response),Ypred_train))
    print(train_r2)

plt.figure(0)
plt.plot(alphas,train_r2)
plt.title('Training')    
plt.show()
