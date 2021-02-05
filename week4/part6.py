import sys
import os
from skimage import io
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from sklearn.ensemble import RandomForestClassifier

Training_slice_path = 'C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Tumor\\*.tif'
Train_mask_path = 'C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Mask\\*.tif'
Test_slices_path = 'C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Test_Tumor\\*.tif'
Test_mask_path = 'C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Test_Mask\\*.tif'

Train_slices = np.array(io.ImageCollection(Training_slice_path))
Train_mask = np.array(io.ImageCollection(Train_mask_path))
Test_slices = np.array(io.ImageCollection(Test_slices_path))
Test_mask = np.array(io.ImageCollection(Test_mask_path))



train_stack = np.stack((Train_slices[:,:,:,0],Train_slices[:,:,:,1],Train_slices[:,:,:,2],Train_mask), axis=3)
train_patch=[]
for i in range (46):
    train_patch.append(extract_patches_2d(train_stack[i,:,:,:], (5, 5), max_patches=2000))

train_patch=np.array(train_patch)
print(train_patch.shape)

X=[]
Y1=[]
for i in range (46):
    X.append(train_patch[i,:,:,:,:-1])
    Y1.append(train_patch[i,:,:,:,-1:])

X=np.array(X)
print(X.shape)
Y1=np.array(Y1)
print(Y1.shape)

X = np.einsum('mijkl->ijklm', X)
Y1 = np.einsum('mijkl->ijklm', Y1)
print(X.shape)

X = np.reshape(X,(2000,3450))
Y1 = np.reshape(Y1,(2000,1150))
# X = train_patch[:,:,:,:-256]
# print(X.shape)

# Y1 = train_patch[:,:,:,-256:]
# print(Y1.shape)

# X = np.reshape(X,(2000,19200))
# Y1 = np.reshape(Y1,(2000,6400))

# Y = []
# for i in range (2000):
#     if np.sum(Y1[i][:]/(6400))>=255/2:
#         Y.append(1)
#     else:
#         Y.append(0)



# Y = []
# for i in range (2000):
#     if np.sum(Y1[i][:]/(25))>=255/2:
#         Y.append(1)
#     else:
#         Y.append(0)

# RFC = RandomForestClassifier(n_estimators=100,min_samples_split=5).fit(X, Y)