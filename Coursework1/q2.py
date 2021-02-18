import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from gradient_descent import gradient_descent  

#a


traindata = pd.read_csv('ADNI_CSF.csv')
y = traindata['TAU']
x = traindata['ABETA']
xmin, xmax = min(x),max(x)
ymin, ymax = min(y),max(y)
xd = np.array([xmin, xmax])
plt.figure(0)
iteration = np.array([0,50,100,10000])
for ite in iteration:
    beta = gradient_descent(y, x, beta=[-1,0.5], alpha = 1, epsilon=1e-10, maxiter=ite)
    print(beta)


    
    yd = beta[1]*xd + beta[0]

    plt.plot(xd, yd, 'k', lw=1, ls='--')

    plt.scatter(x,y)
plt.show()