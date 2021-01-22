import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

traindata = pd.read_csv('adni_adas13_train.csv')
print(traindata)
testdata = pd.read_csv('adni_adas13_train.csv')