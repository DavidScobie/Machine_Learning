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

#b

traindata = pd.read_csv('ADNI_CW1_TRAIN.csv')
testdata = pd.read_csv('ADNI_CW1_TEST.csv')

#seperate y and x into separate dataframes
response = traindata['BRAAK34_SUVR']
test_response = testdata['BRAAK34_SUVR']

column_headers = list(traindata)
inputs_list = column_headers[5:]
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
print('hi')    

#optimise linear, polynomial and RBF SVR's with cross validation
no_Cs = 50
folds = 10
seas = np.logspace(-6,0,base=10,num=no_Cs)
lin_cv=[]
poly_cv=[]
rbf_cv=[]
for sea in seas:
    lin_regr_CV = SVR(kernel='linear',C=sea)
    lin_regr_CV.fit(np.array(scale_feat),np.array(response))
    lin_cv.append(cross_val_score(lin_regr_CV,np.array(scale_feat),np.array(response), cv=folds, scoring='r2'))   
    print('n')
    poly_regr_CV = SVR(kernel='poly',degree=3,C=sea)
    poly_regr_CV.fit(np.array(scale_feat),np.array(response))
    poly_cv.append(cross_val_score(poly_regr_CV,np.array(scale_feat),np.array(response), cv=folds, scoring='r2'))  

    rbf_regr_CV = SVR(kernel='rbf',C=sea)
    rbf_regr_CV.fit(np.array(scale_feat),np.array(response))
    rbf_cv.append(cross_val_score(rbf_regr_CV,np.array(scale_feat),np.array(response), cv=folds, scoring='r2'))  


avg_lin_cv=[]
avg_poly_cv=[]
avg_rbf_cv=[]
for i in range (no_Cs):
    avg_lin_cv.append(np.sum(lin_cv[i]/folds))
    avg_poly_cv.append(np.sum(poly_cv[i]/folds))
    avg_rbf_cv.append(np.sum(rbf_cv[i]/folds))

fig0 = plt.figure(0)
ax = plt.subplot(111)
ax.plot(seas,avg_lin_cv,label='linear')
ax.plot(seas,avg_poly_cv,label='polynomial')
ax.plot(seas,avg_rbf_cv,label='rbf')
plt.xscale('log') 
plt.xlabel('C')
plt.ylabel('performance')
plt.title('Kernel performance')
ax.legend()


#train linear, poly and RBF kernels SVR with different C's, and apply training data to it to get performance plot.
no_Cs = 10
seas = np.logspace(-6,0,base=10,num=no_Cs)
R2_lin = []
R2_poly = []
R2_RBF = []

for sea in seas:

    #linear kernel
    lin_regr = SVR(kernel='linear',C=sea)
    lin_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_lin_train=[]
    rowy=[]

    for i in range (len(traindata)):
        for j in range (len(inputs_list)):
            rowy.append([scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_lin_train.append(lin_regr.predict([flat_list]))
        rowy=[]

    R2_lin.append(r2_score(np.array(response),np.array(Ypred_lin_train)))
    print('what')
    #polynomial kernel
    poly_regr = SVR(kernel='poly',degree=3,C=sea)
    poly_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_poly_train=[]
    rowy=[]

    for i in range (len(traindata)):
        for j in range (len(inputs_list)):
            rowy.append([scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_poly_train.append(poly_regr.predict([flat_list]))
        rowy=[]

    R2_poly.append(r2_score(np.array(response),np.array(Ypred_poly_train)))

    #RBF kernel
    RBF_regr = SVR(kernel='rbf',C=sea)
    RBF_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_RBF_train=[]
    rowy=[]
    print('loop')
    for i in range (len(traindata)):
        for j in range (len(inputs_list)):
            rowy.append([scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_RBF_train.append(RBF_regr.predict([flat_list]))
        rowy=[]

    R2_RBF.append(r2_score(np.array(response),np.array(Ypred_RBF_train)))

fig1 = plt.figure(1)
ax = plt.subplot(111)
ax.plot(seas,R2_lin,label='linear')
ax.plot(seas,R2_poly,label='polynomial')
ax.plot(seas,R2_RBF,label='rbf')
plt.xscale('log') 
plt.xlabel('C')
plt.ylabel('performance')
plt.title('Training data performance')
ax.legend()
print('ho')

#train linear, poly and RBF kernels SVR with different C's, and apply test data to it to get performance plot.
no_Cs = 10
seas = np.logspace(-6,0,base=10,num=no_Cs)
R2_lin_test = []
R2_poly_test = []
R2_RBF_test = []
for sea in seas:

    #linear kernel
    lin_regr = SVR(kernel='linear',C=sea)
    lin_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_lin_test=[]
    rowy=[]
    print('njds')
    for i in range (len(testdata)):
        for j in range (len(inputs_list)):
            rowy.append([test_scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_lin_test.append(lin_regr.predict([flat_list]))
        rowy=[]

    R2_lin_test.append(r2_score(np.array(test_response),np.array(Ypred_lin_test)))

    #polynomial kernel
    poly_regr = SVR(kernel='poly',degree=3,C=sea)
    poly_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_poly_test=[]
    rowy=[]

    for i in range (len(testdata)):
        for j in range (len(inputs_list)):
            rowy.append([test_scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_poly_test.append(poly_regr.predict([flat_list]))
        rowy=[]

    R2_poly_test.append(r2_score(np.array(test_response),np.array(Ypred_poly_test)))

    #RBF kernel
    RBF_regr = SVR(kernel='rbf',C=sea)
    RBF_regr.fit(np.array(scale_feat),np.array(response))
    Ypred_RBF_test=[]
    rowy=[]

    for i in range (len(testdata)):
        for j in range (len(inputs_list)):
            rowy.append([test_scale_feat[inputs_list[j]][i]])
        flat_list = [item for sublist in rowy for item in sublist]
        Ypred_RBF_test.append(RBF_regr.predict([flat_list]))
        rowy=[]

    R2_RBF_test.append(r2_score(np.array(test_response),np.array(Ypred_RBF_test)))



fig2 = plt.figure(2)
ax = plt.subplot(111)
ax.plot(seas,R2_lin_test,label='linear')
ax.plot(seas,R2_poly_test,label='polynomial')
ax.plot(seas,R2_RBF_test,label='rbf')
plt.xscale('log') 
plt.xlabel('C')
plt.ylabel('performance')
plt.title('Test data performance')
ax.legend()
plt.show()







# #train a polynomial kernel SVR
# poly_regr = SVR(kernel='poly',degree=3)
# poly_regr.fit(np.array(scale_feat),np.array(response))

# Ypred_poly_train=[]
# rowy=[]
# for i in range (len(traindata)):
#     for j in range (len(inputs_list)):
#         rowy.append([scale_feat[inputs_list[j]][i]])
#     flat_list = [item for sublist in rowy for item in sublist]
#     Ypred_poly_train.append(poly_regr.predict([flat_list]))
#     rowy=[]

# traindata['YPred_poly']=Ypred_poly_train
# # print(traindata['YPred_poly'])

# #train an rbf kernel SVR
# rbf_regr = SVR(kernel='rbf')
# rbf_regr.fit(np.array(scale_feat),np.array(response))

# Ypred_rbf_train=[]
# rowy=[]
# for i in range (len(traindata)):
#     for j in range (len(inputs_list)):
#         rowy.append([scale_feat[inputs_list[j]][i]])
#     flat_list = [item for sublist in rowy for item in sublist]
#     Ypred_rbf_train.append(rbf_regr.predict([flat_list]))
#     rowy=[]

# traindata['YPred_rbf']=Ypred_rbf_train
# print(traindata['YPred_rbf'])

