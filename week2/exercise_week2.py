import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

traindata = pd.read_csv('adni_adas13_train.csv')

testdata = pd.read_csv('adni_adas13_test.csv')
# print(traindata.head(n=5))
# print(traindata.describe())

response = traindata['ADAS13']
# print(response)

features = traindata[['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]



scaler = StandardScaler(with_mean=True, with_std=1).fit(features).transform(features)



dataset = pd.DataFrame({'AGE': scaler[:, 0], 'Ventricles': scaler[:, 1], 'Hippocampus': scaler[:, 2], 'Entorhinal': scaler[:, 3], 'Fusiform': scaler[:, 4], 'APOE4': scaler[:, 5], 'FDG': scaler[:, 6], 'AV45': scaler[:, 7]})


scale_feat = pd.DataFrame()
feat_heads = ['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']
for feature in feat_heads:
    scaler = StandardScaler(with_mean=features[[feature]].mean, with_std=features[[feature]].std).fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,180)[0]

print(scale_feat.describe())
print(scale_feat)

reg = LinearRegression().fit(scale_feat,response)
print(reg.score(scale_feat,response))
print(reg.coef_)


fig, axs = plt.subplots(3, 3)
axs[0,0].plot(scale_feat['AGE'],response,'o', color='black')
# plt.plot(scale_feat['AGE'],response,'o', color='black')
axs[0,1].plot(scale_feat['Ventricles'],response,'o', color='black')
# plt.plot(scale_feat['Ventricles'],response,'o', color='black')
axs[0,2].plot(scale_feat['Hippocampus'],response,'o', color='black')
# axs.set_aspect('equal')
plt.show()
