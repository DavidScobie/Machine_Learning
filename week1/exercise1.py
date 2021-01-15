import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np


data = pd.read_csv('mixture.csv')
# print(data)
testdata = pd.read_csv('mixture_test.csv')
       

DaFr = data.plot.scatter(x='X1', y='X2', c='Y', colormap='viridis')

# plt.scatter(data.X1, data.X2, c=data.Y)
# plt.show()

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
# distances, indices = nbrs.kneighbors(data)

coords=data[['X1', 'X2']]

#testdata=[testdata.iloc[0,0], testdata.iloc[0,1]]


# test_point=testdata.iloc[0,0]
# test_point=[1,1]
def nearestneighbor(data,testdata):
    dist=[]
    for i in range (len(data)):
        xdist=testdata[0]-data.iloc[i,0]
        ydist=testdata[1]-data.iloc[i,1]
        dist.append((((xdist)**2)+((ydist)**2))**0.5)

    mins=dist.index(min(dist))

    if data['Y'][mins] == 0:
        # print('class 1')
        b=0
    else:
        # print('class 2')
        b=1
    
    return b
# print(dist)    

# smalldata=testdata.iloc[0:6830,:]


Y=[]
for index,row in testdata.iterrows():
# for index,row in smalldata.iterrows():
    Y.append(int(nearestneighbor(data, [row['X1'],row['X2']])))
    # testdata['y'] = nearestneighbor(data, [row['X1'],row['X2']])
testdata['Y']=Y
# smalldata['Y']=Y
testdata.plot.scatter(x='X1', y='X2', c='Y', colormap='viridis')
# smalldata.plot.scatter(x='X1', y='X2', c='Y', colormap='viridis')
# print(smalldata) 
plt.show()
#

