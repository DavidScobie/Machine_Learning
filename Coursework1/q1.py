import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

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

covar = np.cov(np.transpose(CN_both_features))
cov_inv = inv(covar)

#orthogonal vectors to the boundaries
CN_Dementia_orth = np.matmul(cov_inv,np.transpose(CN_mu-Dementia_mu))
CN_MCI_orth = np.matmul(cov_inv,np.transpose(CN_mu-MCI_mu))
Dementia_MCI_orth = np.matmul(cov_inv,np.transpose(Dementia_mu-MCI_mu))
# print(CN_Dementia_orth)

#finding midpoints
CN_Dementia_mid = (CN_mu+Dementia_mu)/2
CN_MCI_mid = (CN_mu+MCI_mu)/2
Dementia_MCI_mid = (Dementia_mu+MCI_mu)/2

#gradient and y intercept
CN_Dementia_grad = CN_Dementia_orth[0]/-CN_Dementia_orth[1]
CN_Dementia_c = CN_Dementia_mid[0][1] - (CN_Dementia_grad*CN_Dementia_mid[0][0])

CN_MCI_grad = CN_MCI_orth[0]/-CN_MCI_orth[1]
CN_MCI_c = CN_MCI_mid[0][1] - (CN_MCI_grad*CN_MCI_mid[0][0])

Dementia_MCI_grad = Dementia_MCI_orth[0]/-Dementia_MCI_orth[1]
Dementia_MCI_c = Dementia_MCI_mid[0][1] - (Dementia_MCI_grad*Dementia_MCI_mid[0][0])

print(CN_Dementia_c)
print(CN_MCI_c)
print(Dementia_MCI_c)

xd = np.array([0, 10])

print(Dementia_MCI_grad)
print(CN_MCI_grad)
print(CN_Dementia_grad)

#equations of boundaries
y_CN_Dementia = CN_Dementia_grad*xd + CN_Dementia_c
y_CN_MCI = CN_MCI_grad*xd + CN_MCI_c
y_Dementia_MCI = Dementia_MCI_grad*xd + Dementia_MCI_c

plt.figure(2)

plt.plot(xd, y_CN_Dementia, 'k', lw=1, ls='--')
plt.plot(xd, y_CN_MCI, 'k', lw=1, ls='--')
plt.plot(xd, y_Dementia_MCI, 'k', lw=1, ls='--')

plt.scatter(CN_both_features[:,0],CN_both_features[:,1])
plt.scatter(Dementia_both_features[:,0],Dementia_both_features[:,1])
plt.scatter(MCI_both_features[:,0],MCI_both_features[:,1])

#d
print(np.max(traindata['ABETA']))
print(np.max(traindata['TAU']))

CN_ABETA_TAU = CN_abeta * CN_tau
DEMENTIA_ABETA_TAU = Dementia_abeta * Dementia_tau
MCI_ABETA_TAU = MCI_abeta * MCI_tau

mean_CN_ABETA_TAU = CN_only.ABETA.mean() * CN_only.TAU.mean()
mean_DEMENTIA_ABETA_TAU = Dementia_only.ABETA.mean() * Dementia_only.TAU.mean()
mean_MCI_ABETA_TAU = MCI_only.ABETA.mean() * MCI_only.TAU.mean()

# x data
CN_3_features = np.transpose(np.array([CN_abeta,CN_tau,CN_ABETA_TAU]))
Dementia_3_features = np.transpose(np.array([Dementia_abeta,Dementia_tau,DEMENTIA_ABETA_TAU]))
MCI_3_features = np.transpose(np.array([MCI_abeta,MCI_tau,MCI_ABETA_TAU]))

# mu
CN_mu = np.array([mean_CN_ABETA,mean_CN_TAU,mean_CN_ABETA_TAU])[np.newaxis]
Dementia_mu = np.array([mean_Dementia_ABETA,mean_Dementia_TAU,mean_DEMENTIA_ABETA_TAU])[np.newaxis]
MCI_mu = np.array([mean_MCI_ABETA,mean_MCI_TAU,mean_MCI_ABETA_TAU])[np.newaxis]

covar = np.cov(np.transpose(CN_3_features))
cov_inv = inv(covar)

#orthogonal vectors to the boundaries
CN_Dementia_orth = np.matmul(cov_inv,np.transpose(CN_mu-Dementia_mu))
CN_MCI_orth = np.matmul(cov_inv,np.transpose(CN_mu-MCI_mu))
Dementia_MCI_orth = np.matmul(cov_inv,np.transpose(Dementia_mu-MCI_mu))
print(Dementia_MCI_orth)

#finding midpoints
CN_Dementia_mid = (CN_mu+Dementia_mu)/2
CN_MCI_mid = (CN_mu+MCI_mu)/2
Dementia_MCI_mid = (Dementia_mu+MCI_mu)/2

#plotting CN Dementia
points = [[-12.4, 1.7, 6.4],
           [-1.9, 1.5, -1.7],
           [4.4, -1.9, -12.4]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt3d.set(xlabel='ABETA', ylabel='TAU', zlabel='ABETA*TAU')

#plotting CN MCI
points = [[-2.6, 0.1, 4],
           [-0.3, 1.5, -0.1],
           [4.4, -0.3, -2.6]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

plt3d.plot_surface(xx, yy, z)

#plotting Dementia MCI
points = [[9.8, -1.6, 4],
           [1.6, 1.5, 1.6],
           [4.4, 1.6, 9.8]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

plt3d.plot_surface(xx, yy, z)

#plotting the underlying points
plt3d.scatter(CN_3_features[:,0],CN_3_features[:,1],CN_3_features[:,2])
plt3d.scatter(Dementia_3_features[:,0],Dementia_3_features[:,1],Dementia_3_features[:,2])
plt3d.scatter(MCI_3_features[:,0],MCI_3_features[:,1],MCI_3_features[:,2])

plt.show()











