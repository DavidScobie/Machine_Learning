import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

#read training data in
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
# plt.show()


MCI_less.reset_index(drop=True,inplace=True)
print(list(MCI_less['DX']))
response = []

for i in range (len(MCI_less)):
# for index, row  in MCI_less.iterrows():
    # print(row)
    if (MCI_less['DX'][i]) == 'CN':
        response.append(0)
    else:
        response.append(1)
print(response)
#normailse training data
features = MCI_less[['ABETA','TAU']]

reg = LinearRegression().fit(features,response)
print(reg.coef_)




