import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

traindata = pd.read_csv('mixture.csv')
testdata = pd.read_csv('mixture_test.csv')

#seperate y and x into separate dataframes
response = traindata['Y']
features = traindata[['X1','X2']]

#normailse training data
scale_feat = pd.DataFrame()
feat_heads = ['X1','X2']
for feature in feat_heads:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,200)[0]

reg = LinearRegression().fit(scale_feat,response)
w1,w2 = reg.coef_
b = reg.intercept_

#normalise test data
test_features = testdata[['X1','X2']]
test_scale_feat = pd.DataFrame()
for feature in feat_heads:
    test_scaler = StandardScaler()
    test_scaler = test_scaler.fit(test_features[[feature]])
    test_scaler = test_scaler.transform(test_features[[feature]])
    test_scale_feat[feature] = test_scaler.reshape(1,6831)[0] 

#Apply test data to the trained linear regression model
Ypred=[]
for i in range (len(testdata)):
    Ypred.append(reg.predict(np.array([[test_scale_feat['X1'][i],test_scale_feat['X2'][i]]])))
testdata['YPred']=Ypred

Boolean = (testdata['YPred'] > 0.5)
testdata['YPred'][Boolean == True] = 1
testdata['YPred'][Boolean == False] = 0

# Calculate the intercept and gradient of the decision boundary.
c = (0.5 - b)/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
xmin, xmax = min(test_scale_feat['X1']),max(test_scale_feat['X1'])
ymin, ymax = min(test_scale_feat['X2']),max(test_scale_feat['X2'])
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.ylabel('X2_scaled')
plt.xlabel('X1_scaled')
plt.scatter(test_scale_feat['X1'],test_scale_feat['X2'], c=testdata['YPred'])


traindata2 = pd.read_csv('adni_conversion_train.csv')
response2 = traindata2['conversion']

#normailse training data 2
train2_features = traindata2[['CDRSB','ADAS13','MMSE','MOCA','AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]
train2_featheads = ['CDRSB','ADAS13','MMSE','MOCA','AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']
train2_scale_feat = pd.DataFrame()
for feature in train2_featheads:
    # train2_scaler = StandardScaler(with_mean=train2_features[[feature]].mean, with_std=train2_features[[feature]].std).fit(train2_features[[feature]]).transform(train2_features[[feature]])
    train2_scaler = StandardScaler().fit(train2_features[[feature]]).transform(train2_features[[feature]])
    train2_scale_feat[feature] = train2_scaler.reshape(1,145)[0]

reg = LogisticRegression(penalty='elasticnet',solver='saga',C=1,l1_ratio=0.5).fit(train2_scale_feat,response2)
print(reg.coef_)

reg_CV = LogisticRegressionCV(penalty='elasticnet',solver='saga',scoring = 'balanced_accuracy', l1_ratios = [0.5], Cs =100).fit(train2_scale_feat,response2)
print(reg_CV.coef_)
print(reg_CV.Cs_)
print(reg_CV.C_)

testdata2 = pd.read_csv('adni_conversion_test.csv')

#normailse test data 2
test2_features = testdata2[['CDRSB','ADAS13','MMSE','MOCA','AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]
test2_scale_feat = pd.DataFrame()
for feature in train2_featheads:
    # test2_scaler = StandardScaler(with_mean=test2_features[[feature]].mean, with_std=test2_features[[feature]].std).fit(test2_features[[feature]]).transform(test2_features[[feature]])
    test2_scaler = StandardScaler().fit(test2_features[[feature]]).transform(test2_features[[feature]])
    test2_scale_feat[feature] = test2_scaler.reshape(1,143)[0]

#Apply test data to the trained logistic regression model
Ypred=[]
for i in range (len(testdata2)):
    Ypred.append(reg_CV.predict(np.array([[test2_scale_feat['CDRSB'][i],test2_scale_feat['ADAS13'][i],test2_scale_feat['MMSE'][i],test2_scale_feat['MOCA'][i],test2_scale_feat['AGE'][i],test2_scale_feat['Ventricles'][i],test2_scale_feat['Hippocampus'][i],test2_scale_feat['Entorhinal'][i],test2_scale_feat['Fusiform'][i],test2_scale_feat['APOE4'][i],test2_scale_feat['FDG'][i],test2_scale_feat['AV45'][i]]])))
testdata2['YPred']=Ypred

print(testdata2)

print(confusion_matrix(testdata2['conversion'].astype('float64'), testdata2['YPred'].astype('float64')))

print(roc_auc_score(testdata2['conversion'].astype('float64'),testdata2['YPred'].astype('float64')))

print(balanced_accuracy_score(testdata2['conversion'].astype('float64'),testdata2['YPred'].astype('float64')))

plt.show()


