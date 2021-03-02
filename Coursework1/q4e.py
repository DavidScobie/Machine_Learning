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

#normailse test data
test_features = testdata[inputs_list]
test_scale_feat = pd.DataFrame()
for feature in inputs_list:
    scaler = StandardScaler()
    scaler = scaler.fit(test_features[[feature]]).transform(test_features[[feature]])
    test_scale_feat[feature] = scaler.reshape(1,len(testdata['BRAAK34_SUVR']))[0]

# response: BRAAK34_SUVR
cv_outer = KFold(n_splits=10)
outer_results = list()
for train_ix, test_ix in cv_outer.split(scale_feat):
	# split data

    X_train = scale_feat.loc[train_ix,inputs_list[:]]
    X_test = scale_feat.loc[test_ix,inputs_list[:]]
    y_train = response[train_ix]
    y_test = response[test_ix]

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=2)
	# define the model
    model = SVR()
	# define search space
    space = dict()
    space['kernel'] = ['linear','poly','rbf']
    space['C'] = np.logspace(-6,0,base=10,num=5)
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

best_SVR_34 = SVR(kernel='linear',C=0.001)
best_SVR_34.fit(np.array(scale_feat),np.array(response))
Ypred_34_test=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_34_test.append(best_SVR_34.predict([flat_list]))
    rowy=[]

R2_34_test = r2_score(np.array(test_response),np.array(Ypred_34_test))
print(R2_34_test)

# response: BRAAK1_SUVR
response = traindata['BRAAK1_SUVR']
test_response = testdata['BRAAK1_SUVR']
cv_outer = KFold(n_splits=10)
outer_results = list()
for train_ix, test_ix in cv_outer.split(scale_feat):
	# split data

    X_train = scale_feat.loc[train_ix,inputs_list[:]]
    X_test = scale_feat.loc[test_ix,inputs_list[:]]
    y_train = response[train_ix]
    y_test = response[test_ix]

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=2)
	# define the model
    model = SVR()
	# define search space
    space = dict()
    space['kernel'] = ['linear','poly','rbf']
    space['C'] = np.logspace(-6,0,base=10,num=5)
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

best_SVR_1 = SVR(kernel='rbf',C=1)
best_SVR_1.fit(np.array(scale_feat),np.array(response))
Ypred_1_test=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_1_test.append(best_SVR_1.predict([flat_list]))
    rowy=[]

R2_1_test = r2_score(np.array(test_response),np.array(Ypred_1_test))
print(R2_1_test)

# response: BRAAK2_SUVR
response = traindata['BRAAK2_SUVR']
test_response = testdata['BRAAK2_SUVR']
cv_outer = KFold(n_splits=10)
outer_results = list()
for train_ix, test_ix in cv_outer.split(scale_feat):
	# split data

    X_train = scale_feat.loc[train_ix,inputs_list[:]]
    X_test = scale_feat.loc[test_ix,inputs_list[:]]
    y_train = response[train_ix]
    y_test = response[test_ix]

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=2)
	# define the model
    model = SVR()
	# define search space
    space = dict()
    space['kernel'] = ['linear','poly','rbf']
    space['C'] = np.logspace(-6,0,base=10,num=5)
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


best_SVR_2 = SVR(kernel='linear',C=0.001)
best_SVR_2.fit(np.array(scale_feat),np.array(response))
Ypred_2_test=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_2_test.append(best_SVR_2.predict([flat_list]))
    rowy=[]

R2_2_test = r2_score(np.array(test_response),np.array(Ypred_2_test))
print(R2_2_test)

# response: BRAAK56_SUVR
response = traindata['BRAAK56_SUVR']
test_response = testdata['BRAAK56_SUVR']
cv_outer = KFold(n_splits=10)
outer_results = list()
for train_ix, test_ix in cv_outer.split(scale_feat):
	# split data

    X_train = scale_feat.loc[train_ix,inputs_list[:]]
    X_test = scale_feat.loc[test_ix,inputs_list[:]]
    y_train = response[train_ix]
    y_test = response[test_ix]

    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=2)
	# define the model
    model = SVR()
	# define search space
    space = dict()
    space['kernel'] = ['linear','poly','rbf']
    space['C'] = np.logspace(-6,0,base=10,num=5)
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

best_SVR_56 = SVR(kernel='linear',C=0.001)
best_SVR_56.fit(np.array(scale_feat),np.array(response))
Ypred_56_test=[]
rowy=[]
for i in range (len(testdata)):
    for j in range (len(inputs_list)):
        rowy.append([test_scale_feat[inputs_list[j]][i]])
    flat_list = [item for sublist in rowy for item in sublist]
    Ypred_56_test.append(best_SVR_56.predict([flat_list]))
    rowy=[]

R2_56_test = r2_score(np.array(test_response),np.array(Ypred_56_test))
print(R2_56_test)