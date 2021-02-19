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
norm_resp = np.round(response/np.max(response),0)

column_headers = list(traindata)
inputs_list = column_headers[4:]
features = traindata[inputs_list]

#normailse training data
scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(features[[feature]]).transform(features[[feature]])
    scale_feat[feature] = scaler.reshape(1,len(response))[0]


regr = ElasticNet(random_state=0,l1_ratio=0.5)
regr.fit(np.array(scale_feat),np.array(response))
# print(regr.coef_)

regrCV = ElasticNetCV(cv=10,l1_ratio=0.5)
regrCV.fit(np.array(scale_feat),np.array(response))
print(regrCV.alpha_)



# reg = LogisticRegression(penalty='elasticnet',solver='saga',C=1,l1_ratio=0.5).fit(scale_feat,norm_resp)
# print(reg.coef_)
# print(reg.intercept_)
# print(reg.n_iter_)
# print(np.max(np.round(response,0)))
# print(np.min(np.round(response,0)))


# reg_CV = LogisticRegressionCV(penalty='elasticnet',solver='saga',scoring = 'balanced_accuracy', l1_ratios = [0.5], Cs =10, cv=10).fit(scale_feat,norm_resp)
# print(reg_CV.coef_)
# print(reg_CV.Cs_)
# print(reg_CV.C_)
# print(reg_CV.n_iter_)
# prediction = reg_CV.predict(scale_feat)
# print(prediction)
# R2 = r2_score(norm_resp, prediction)
# print(R2)

