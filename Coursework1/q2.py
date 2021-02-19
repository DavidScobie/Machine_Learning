import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from gradient_descent import gradient_descent  

#b


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

#c

plt.figure(1)
# Lambdas = np.array([0,1000,100000])
Lambdas = np.array([0,1000])
for Lambda in Lambdas:
    beta = gradient_descent(y, x, beta=[-1,0.5], alpha = 1, epsilon=1e-10, maxiter=10000, Lam=Lambda)
    print(beta)

    yd = beta[1]*xd + beta[0]

    plt.plot(xd, yd, 'k', lw=1, ls='--')

    plt.scatter(x,y)

# plt.figure(2)
# Lambda = 10
# beta = gradient_descent(y, x, beta=[-1,0.5], alpha = 0.011, epsilon=1e-10, maxiter=100, Lam=Lambda)
# print(beta)

# yd = beta[1]*xd + beta[0]

# plt.plot(xd, yd, 'k', lw=1, ls='--')

# plt.scatter(x,y)

plt.figure(3)
Lambda = 100000
beta = gradient_descent(y, x, beta=[-1,0.5], alpha = 1, epsilon=1e-10, maxiter=10000, Lam=Lambda)
print(beta)

yd = beta[1]*xd + beta[0]

plt.plot(xd, yd, 'k', lw=1, ls='--')

plt.scatter(x,y)

plt.show()