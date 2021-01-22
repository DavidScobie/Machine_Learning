import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

traindata = pd.read_csv('adni_adas13_train.csv')

testdata = pd.read_csv('adni_adas13_test.csv')
# print(traindata.head(n=5))
# print(traindata.describe())

response = traindata['ADAS13']
# print(response)

features = traindata[['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']]
# print(features)


scaler = StandardScaler(with_mean=True, with_std=1).fit(features).transform(features)
# print(scaler)
# scaler = StandardScaler(with_mean=0, with_std=1).fit(features['AGE']).transform(features['AGE'])


dataset = pd.DataFrame({'AGE': scaler[:, 0], 'Ventricles': scaler[:, 1], 'Hippocampus': scaler[:, 2], 'Entorhinal': scaler[:, 3], 'Fusiform': scaler[:, 4], 'APOE4': scaler[:, 5], 'FDG': scaler[:, 6], 'AV45': scaler[:, 7]})
# print(dataset.describe())

# smalld = pd.DataFrame({'AGE': scaler[:, 0]})

# scaled = StandardScaler(features, copy=True, with_mean=0, with_std=1)
# print(scaled.fit(features))
# print(scaled.mean_)
# print(scaled.describe())
dataframe = pd.DataFrame()
feat_heads = ['AGE','Ventricles','Hippocampus','Entorhinal','Fusiform','APOE4','FDG','AV45']
for feature in feat_heads:
    scale_feat = StandardScaler(with_mean=features[[feature]].mean, with_std=features[[feature]].std).fit(features[[feature]]).transform(features[[feature]])
    dataframe[feature] = scale_feat.reshape(1,180)[0]

print(dataframe.describe())
print(dataframe)


