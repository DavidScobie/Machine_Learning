import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

#read training data in
traindata = pd.read_csv('ADNI_CSF.csv')
print(traindata.head())