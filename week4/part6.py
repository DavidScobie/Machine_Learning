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

X = np.reshape(X,(2000*46,75))
Y1 = np.reshape(Y1,(2000*46,25))
print(X.shape)

Y = []
for i in range (2000*46):
    if np.sum(Y1[i][:]/(25))>=255/2:
        Y.append(1)
    else:
        Y.append(0)
print(np.array(Y).shape)

RFC = RandomForestClassifier(n_estimators=100,min_samples_split=5).fit(X, Y)

print(Test_slices[1,:,:,:].shape)
padded_image=[]
for i in range (5):
    padded_image.append(np.pad(Test_slices[i,:,:,:], ((2, 2), (2, 2),(0,0)),'constant',constant_values=0))
padded_image = np.array(padded_image)
print(padded_image.shape)

patchy=[]
for i in range (5):
    patchy.append(extract_patches_2d(padded_image[i,:,:,:], (5, 5)))
patchy = np.array(patchy)
patchy = np.reshape(patchy,(65536*5,75))

test_images = RFC.predict(patchy)

Y_test = np.tile(test_images,(75,1))
Y_test = np.swapaxes(Y_test,0,1)
Y_test = np.reshape(Y_test, newshape = (65536*5,5,5,3), order = 'C')

#all good up to here

Y_test_im = np.zeros((256**2,3,5))
for k in range (5):
    for i in range (256**2):
        for j in range (3):
            Y_test_im[i,j,k] = np.average(Y_test[i+(k*(256**2)),:,:,j])

for i in range (5):
    plt.figure(i)
    Y_test_im = np.reshape(Y_test_im, newshape = (256,256,3,5), order = 'C')
    plt.imshow(Y_test_im[:,:,:,i])
plt.show()