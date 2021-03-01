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
inputs_list = column_headers[5:]
features = traindata[inputs_list]

#normailse training data
scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,len(response))[0]

#test data
test_features = testdata[inputs_list]
test_response = testdata['BRAAK34_SUVR']

#normailse test data
test_scale_feat = pd.DataFrame()
for feature in inputs_list:
    test_scaler = StandardScaler()
    test_scaler = test_scaler.fit(test_features[[feature]]).transform(test_features[[feature]])
    test_scale_feat[feature] = test_scaler.reshape(1,len(test_features))[0]


#elastic net
train_r2=[]
test_r2=[]
alphas = np.linspace(0,0.1,num=11)
for alph in alphas:
    regr = ElasticNet(random_state=0,alpha=alph,l1_ratio=0.5)
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
    # print(train_r2)


    # Apply test data to the trained linear regression model

    Ypred_test=[]
    rowy=[]
    for i in range (len(testdata)):
        for j in range (len(inputs_list)):
            rowy.append([test_scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_test.append(regr.predict([flat_list]))
        rowy=[]
        
    testdata['YPred']=Ypred_test
    test_r2.append(r2_score(np.array(test_response),Ypred_test))
    # print(test_r2)

print(test_r2)
# alphas = np.array([0.01,0.02,0.03])
#Cross validation
folds=10
regrCV = ElasticNetCV(cv=folds,l1_ratio=0.5,alphas=alphas)
regrCV.fit(np.array(scale_feat),np.array(response))
print(regrCV.alpha_)

n_mses=[]
for i in range (len(alphas)):
    n_mses.append((np.sum(regrCV.mse_path_[i])/folds)*(len(traindata)))
print(n_mses)

mean_y = np.mean(np.array(response))
y_min_mean=[]
for i in range (len(traindata)):
    y_min_mean.append((mean_y-(response[i]))**2)
sum_y_min_mean = np.sum(y_min_mean)

r_squ=[]
for i in range (len(alphas)):
    r_squ.append(1-(n_mses[i]/sum_y_min_mean))
print(r_squ)

plt.figure(0)
plt.plot(alphas,train_r2)
plt.title('Training')

plt.figure(1)
plt.plot(alphas,test_r2)
plt.title('Testing')

rev_alph = np.linspace(0.1,0,num=11)
plt.figure(2)
plt.plot(rev_alph,r_squ)
plt.title('Cross validation')

#optimised elastic net
bet_regr = ElasticNet(random_state=0,alpha=regrCV.alpha_,l1_ratio=0.5)
bet_regr.fit(np.array(scale_feat),np.array(response))

Ypred_test_2=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_test_2.append(bet_regr.predict([flat_list]))
    rowy=[]
    
bet_test_r2 = r2_score(np.array(test_response),Ypred_test_2)
print(bet_test_r2)

plt.show()










