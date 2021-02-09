import sys
import os
from skimage import io
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from sklearn.ensemble import RandomForestClassifier

slice = io.imread('C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Tumor\\TCGA_CS_6186_20000601_19.tif')
segment = io.imread('C:\\PHD\\Machine_Learning\\week4\\brain_tumor_custom\\Mask\\TCGA_CS_6186_20000601_19_mask.tif')
plt.figure(0)
plt.imshow(slice)
plt.figure(1)
plt.imshow(segment)
together = np.stack((slice[:,:,0],slice[:,:,1],slice[:,:,2],segment), axis=2)
print(together.shape)
patchy = extract_patches_2d(together, (5, 5), max_patches=1000)
print(patchy.shape)
X = patchy[:,:,:,:-1]
print(X.shape)
X = np.reshape(X,(1000,75))
Y1 = patchy[:,:,:,-1:]
Y1 = np.reshape(Y1,(1000,25))
Y = []
for i in range (1000):
    if np.sum(Y1[i][:]/(25))>=255/2:
        Y.append(1)
    else:
        Y.append(0)
print(np.array(Y).shape)
RFC = RandomForestClassifier(n_estimators=100,min_samples_split=5).fit(X, Y)

padded_image = np.pad(slice, ((2, 2), (2, 2),(0,0)),'constant',constant_values=0)
plt.imshow(padded_image)
patchy2 = extract_patches_2d(padded_image, (5, 5))
test_image = RFC.predict_proba(np.reshape(patchy2,(65536,75)))
print(test_image.shape)
Y_test = np.tile(np.transpose(test_image[:,0]),(75,1))
Y_test = np.swapaxes(Y_test,0,1)
Y_test = np.reshape(Y_test, newshape = (65536,5,5,3), order = 'C')

Y_test_im = np.zeros((256**2,3))
for i in range (256**2):
    for j in range (3):
        Y_test_im[i,j] = np.average(Y_test[i,:,:,j])


Y_test_im = np.reshape(Y_test_im, newshape = (256,256,3), order = 'C')
plt.figure(2)
plt.imshow(Y_test_im)

plt.show()
