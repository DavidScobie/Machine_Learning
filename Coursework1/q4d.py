import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy import mean
from numpy import std

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

#elastic net
cv_outer = KFold(n_splits=10, shuffle=True)
outer_results = list()
for train_ix, test_ix in cv_outer.split(scale_feat):
	# split data

    X_train = scale_feat.loc[train_ix,inputs_list[:]]
    X_test = scale_feat.loc[test_ix,inputs_list[:]]
    y_train = response[train_ix]
    y_test = response[test_ix]

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=5, shuffle=True)
	# define the model
    model = ElasticNet()
	# define search space
    space = dict()
    space['alpha'] = np.linspace(0.0001,0.1,num=11)
    space['l1_ratio'] = [0.5]
    # define search
    search = GridSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True)
	# execute search
    result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
    best_model = result.best_estimator_
	# evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
	# evaluate the model
    acc = r2_score(y_test, yhat)
    # store the result
    outer_results.append(acc)
    # report progress
    print('>R2=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('R2_score: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


# #SVR
# cv_outer = KFold(n_splits=10)
# outer_results = list()
# for train_ix, test_ix in cv_outer.split(scale_feat):
# 	# split data

#     X_train = scale_feat.loc[train_ix,inputs_list[:]]
#     X_test = scale_feat.loc[test_ix,inputs_list[:]]
#     y_train = response[train_ix]
#     y_test = response[test_ix]

#     # configure the cross-validation procedure
#     cv_inner = KFold(n_splits=5)
# 	# define the model
#     model = SVR()
# 	# define search space
#     space = dict()
#     space['kernel'] = ['linear','poly','rbf']
#     space['C'] = np.logspace(-6,0,base=10,num=20)
#     # define search
#     search = GridSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True)
# 	# execute search
#     result = search.fit(X_train, y_train)
# 	# get the best performing model fit on the whole training set
#     best_model = result.best_estimator_
# 	# evaluate model on the hold out dataset
#     yhat = best_model.predict(X_test)
# 	# evaluate the model
#     acc = r2_score(y_test, yhat)
#     # store the result
#     outer_results.append(acc)
#     # report progress
#     print('>R2=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# # summarize the estimated performance of the model
# print('R2_score: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


# #Random forest regressor
# cv_outer = KFold(n_splits=10, shuffle=True)
# outer_results = list()
# for train_ix, test_ix in cv_outer.split(scale_feat):
# 	# split data

#     X_train = scale_feat.loc[train_ix,inputs_list[:]]
#     X_test = scale_feat.loc[test_ix,inputs_list[:]]
#     y_train = response[train_ix]
#     y_test = response[test_ix]

#     # configure the cross-validation procedure
#     cv_inner = KFold(n_splits=5, shuffle=True)
# 	# define the model
#     model = RandomForestRegressor()
# 	# define search space
#     space = dict()
#     space['n_estimators'] = [500]
#     space['min_samples_split'] = [5]
#     # define search
#     search = GridSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True)
# 	# execute search
#     result = search.fit(X_train, y_train)
# 	# get the best performing model fit on the whole training set
#     best_model = result.best_estimator_
# 	# evaluate model on the hold out dataset
#     yhat = best_model.predict(X_test)
# 	# evaluate the model
#     acc = r2_score(y_test, yhat)
#     # store the result
#     outer_results.append(acc)
#     # report progress
#     print('>R2=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# # summarize the estimated performance of the model
# print('R2_score: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

