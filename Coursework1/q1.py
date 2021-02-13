import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv

#a


traindata = pd.read_csv('ADNI_CSF.csv')
MCI_less = traindata[traindata.DX != 'MCI']
print(MCI_less)

CN_only = traindata[traindata.DX == 'CN']
mean_CN_ABETA = CN_only.ABETA.mean()
std_CN_ABETA = CN_only.ABETA.std()

Dementia_only = traindata[traindata.DX == 'Dementia']
mean_Dementia_ABETA = Dementia_only.ABETA.mean()
std_Dementia_ABETA = Dementia_only.ABETA.std()

print(mean_CN_ABETA)
print(mean_Dementia_ABETA)
print(std_CN_ABETA)
print(std_Dementia_ABETA)

n=100
x=np.linspace(0,10,num=n)

fx_CN_end = []
fx_CN = []
fx_CN_start = 1/(std_CN_ABETA*(np.sqrt(2*np.pi)))
for i in range (n):
    fx_CN_power = ((x[i]-mean_CN_ABETA)/(std_CN_ABETA))**2
    fx_CN_end.append(np.exp(-0.5*fx_CN_power))

for i in range (n):
    fx_CN.append(fx_CN_start * fx_CN_end[i])

fx_Dementia_end = []
fx_Dementia = []
fx_Dementia_start = 1/(std_Dementia_ABETA*(np.sqrt(2*np.pi)))
for i in range (n):
    fx_Dementia_power = ((x[i]-mean_Dementia_ABETA)/(std_Dementia_ABETA))**2
    fx_Dementia_end.append(np.exp(-0.5*fx_Dementia_power))

for i in range (n):
    fx_Dementia.append(fx_Dementia_start * fx_Dementia_end[i])

a = ((-1/(std_Dementia_ABETA**2))+(1/(std_CN_ABETA**2)))
b = ((2*mean_Dementia_ABETA)/(std_Dementia_ABETA**2))-((2*mean_CN_ABETA)/(std_CN_ABETA**2))
c = (-2*np.log((std_Dementia_ABETA)/(std_CN_ABETA)))-((mean_Dementia_ABETA**2)/(std_Dementia_ABETA**2))+((mean_CN_ABETA**2)/(std_CN_ABETA**2))
x_bound = (-b-np.sqrt((b**2)-(4*a*c)))/(2*a)
print(x_bound)

plt.figure(0)
plt.plot(x,fx_CN)
plt.plot(x,fx_Dementia)
plt.axvline(x=x_bound, ymin=0, ymax=1)

#b

MCI_less.reset_index(drop=True,inplace=True)
response = []

for i in range (len(MCI_less)):
    if (MCI_less['DX'][i]) == 'CN':
        response.append(0)
    else:
        response.append(1)

features = MCI_less[['ABETA','TAU']]

reg = LinearRegression().fit(features,response)
print(reg.coef_)
print(reg.intercept_)

w1,w2 = reg.coef_
b = reg.intercept_

c = (0.5 - b)/w2
m = -w1/w2

print(m)
print(c)

xmin, xmax = min(features['ABETA']),max(features['ABETA'])
ymin, ymax = min(features['TAU']),max(features['TAU'])
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.figure(1)
plt.plot(xd, yd, 'k', lw=1, ls='--')

plt.scatter(MCI_less['ABETA'],MCI_less['TAU'], c=response)
# plt.show()

#c

MCI_only = traindata[traindata.DX == 'MCI']
mean_MCI_ABETA = MCI_only.ABETA.mean()
std_MCI_ABETA = MCI_only.ABETA.std()
print(mean_MCI_ABETA)
print(std_MCI_ABETA)

CN_abeta = np.array(CN_only['ABETA'])
Dementia_abeta = np.array(Dementia_only['ABETA'])
MCI_abeta = np.array(MCI_only['ABETA'])
CN_tau = np.array(CN_only['TAU'])
Dementia_tau = np.array(Dementia_only['TAU'])
MCI_tau = np.array(MCI_only['TAU'])

mean_MCI_ABETA = MCI_only.ABETA.mean()
mean_MCI_TAU = MCI_only.TAU.mean()
mean_CN_TAU = CN_only.TAU.mean()
mean_Dementia_TAU = Dementia_only.TAU.mean()

# x data
CN_both_features = np.transpose(np.array([CN_abeta,CN_tau]))
Dementia_both_features = np.transpose(np.array([Dementia_abeta,Dementia_tau]))
MCI_both_features = np.transpose(np.array([MCI_abeta,MCI_tau]))

# mu
CN_mu = np.array([mean_CN_ABETA,mean_CN_TAU])[np.newaxis]
Dementia_mu = np.array([mean_Dementia_ABETA,mean_Dementia_TAU])[np.newaxis]
MCI_mu = np.array([mean_MCI_ABETA,mean_MCI_TAU])[np.newaxis]

print(CN_both_features.shape)

covar = np.cov(np.transpose(CN_both_features))
cov_inv = inv(covar)

# print(cov_inv.shape)
# print(np.transpose(CN_mu-Dementia_mu))


CN_dementia_orth = np.matmul(cov_inv,np.transpose(CN_mu-Dementia_mu))
print(CN_dementia_orth)

midpoint = (CN_mu+Dementia_mu)/2
print(midpoint)

plt.figure(2)
plt.scatter(CN_both_features[:,0],CN_both_features[:,1])
plt.scatter(Dementia_both_features[:,0],Dementia_both_features[:,1])
plt.show()

# Di_fu_CN_s_end = np.matmul(cov_inv,np.transpose(CN_both_features))
# Di_fu_CN = np.matmul(CN_mu,Di_fu_CN_s_end)

# Di_fu_CN_mid_1 = np.matmul(cov_inv,np.transpose(CN_mu))
# Di_fu_CN_mid_2 = np.matmul(CN_mu,Di_fu_CN_mid_1)
# Di_fu_CN_mid = -0.5*Di_fu_CN_mid_2

# Di_fu_CN_end = np.log(0.33)

# Di_fu_CN = Di_fu_CN - Di_fu_CN_mid + Di_fu_CN_end
# print(Di_fu_CN)
# plt.figure(2)
# plt.plot(Di_fu_CN)
# plt.show()








