import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
'''
data1 = np.load('C:\\PHD\\Machine_Learning\\mphy0041\\tutorials\\segmentation\\result\\label_test00_step000006-tf.npy')
print(data1.shape)
# slice1 = data1[0,:,:]
# print(slice1.shape)
# plt.figure(0)
# plt.imshow(slice1)
# plt.show()

class DataReader:
    def __init__(self, folder_name):
        self.folder_name = folder_name
    def load_images_train(self, indices_mb):
        return self.load_npy_files(["image_train%02d.npy" % idx for idx in indices_mb])
    def load_images_test(self, indices_mb):
        return self.load_npy_files(["image_test%02d.npy" % idx for idx in indices_mb])
    def load_labels_train(self, indices_mb):
        return self.load_npy_files(["label_train%02d.npy" % idx for idx in indices_mb])
    def load_npy_files(self, file_names):
        images = [np.float32(np.load(os.path.join(self.folder_name, fn))) for fn in file_names]
        return np.expand_dims(np.stack(images, axis=0), axis=4)[:, ::2, ::2, ::2, :]

data = DataReader('C:\\PHD\\Machine_Learning\\mphy0041\\tutorials\\segmentation\\data\\datasets-promise12\\')
print(data.load_images_train(range(4)).shape)
'''

A = np.random.rand(3,3,3,32,32,2)
img = tf.random.normal((4,16,64,64,32))

y = tf.nn.conv3d(img, A[..., 0], strides= [1,1,1,1,1], padding= 'SAME')
print(y.shape)

