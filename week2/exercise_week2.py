import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

#read training and test data in
traindata = pd.read_csv('adni_adas13_train.csv')
testdata = pd.read_csv('adni_adas13_test.csv')

#seperate y and x into separate dataframes
response = traindata['ADAS13']
features = traindata[['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]

#normailse training data
scale_feat = pd.DataFrame()
feat_heads = ['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']
for feature in feat_heads:
    scaler = StandardScaler(with_mean=features[[feature]].mean, with_std=features[[feature]].std).fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,180)[0]

#train the linear regression on training data
reg = LinearRegression().fit(scale_feat,response)
# print(reg.score(scale_feat,response))
print(reg.coef_)
# print(reg.intercept_)
# print(reg.predict(np.array([[3, 0,0,0,0,0,0,0]])))

#normalise test data
test_features = testdata[['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]
test_scale_feat = pd.DataFrame()
for feature in feat_heads:
    test_scaler = StandardScaler(with_mean=test_features[[feature]].mean, with_std=test_features[[feature]].std).fit(test_features[[feature]]).transform(test_features[[feature]])
    test_scale_feat[feature] = test_scaler.reshape(1,180)[0] 

#Apply test data to the trained linear regression model
Ypred=[]
for i in range (len(testdata)):
    Ypred.append(reg.predict(np.array([[test_scale_feat['AGE'][i],test_scale_feat['Ventricles'][i],test_scale_feat['Hippocampus'][i],test_scale_feat['Entorhinal'][i],test_scale_feat['Fusiform'][i],test_scale_feat['APOE4'][i],test_scale_feat['FDG'][i],test_scale_feat['AV45'][i]]])))
testdata['YPred']=Ypred

#Plotting some features against y for the training data
# # fig, axs = plt.subplots(3, 3)
# # axs[0,0].plot(scale_feat['AGE'],response,'o', color='black')
# # axs[0,0].set(xlabel='AGE', ylabel='y')
# # axs[0,1].plot(scale_feat['Ventricles'],response,'o', color='black')
# # axs[0,1].set(xlabel='Ventricles', ylabel='y')
# # axs[0,2].plot(scale_feat['Hippocampus'],response,'o', color='black')
# # axs[0,2].set(xlabel='Hippocampus', ylabel='y')

#linear regression correlation and MSE
print(np.corrcoef(testdata['ADAS13'].to_numpy().astype(float),testdata['YPred'].to_numpy().astype(float)))
print(mean_squared_error(testdata['ADAS13'].to_numpy().astype(float),testdata['YPred'].to_numpy().astype(float)))

#Ridge regression on the training data
rij = Ridge(alpha=10000).fit(scale_feat,response)
print(rij.coef_)

#Applying test data to the trained ridge regression model
Ypred_rij=[]
for i in range (len(testdata)):
    Ypred_rij.append(rij.predict(np.array([[test_scale_feat['AGE'][i],test_scale_feat['Ventricles'][i],test_scale_feat['Hippocampus'][i],test_scale_feat['Entorhinal'][i],test_scale_feat['Fusiform'][i],test_scale_feat['APOE4'][i],test_scale_feat['FDG'][i],test_scale_feat['AV45'][i]]])))
testdata['YPred_ridge']=Ypred_rij

#Ridge regression correlation and MSE
print(np.corrcoef(testdata['ADAS13'].to_numpy().astype(float),testdata['YPred_ridge'].to_numpy().astype(float)))
print(mean_squared_error(testdata['ADAS13'].to_numpy().astype(float),testdata['YPred_ridge'].to_numpy().astype(float)))

#Scatterplot of observed and predicted for test dataset
plt.plot(testdata['ADAS13'],testdata['YPred'],'o', color='black')
plt.ylabel('prdeicted')
plt.xlabel('observed')
plt.show()