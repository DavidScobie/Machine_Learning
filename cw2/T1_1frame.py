import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import h5py

num_subjects = 1
filename = './data/dataset70-200.h5'
subject_indicies = range(num_subjects)

#I think this needs to be by 3 (or maybe 4) as we have a stack of 1 frame and 3 labels in the generator 
frame_size = np.array([52,58,1])

#image a slice
frame1 = tf.transpose(tf.keras.utils.HDF5Matrix(filename, 'label_0000_000_02' )) / 255
img = tf.image.convert_image_dtype(frame1, tf.float32)
plt.imshow(img)

##MY DATA GENERATOR
def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        idx_frame_indics = range(num_subjects)
        for idx_frame in idx_frame_indics:
            f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
            frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
            l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
            label0 = tf.keras.utils.HDF5Matrix(filename, l0_dataset)
            yield(tf.expand_dims(frame, axis=2), tf.expand_dims(label0, axis=2))

dataset = tf.data.Dataset.from_generator(generator = my_data_generator, 
                                         output_types = (tf.float32, tf.float32),
                                         output_shapes = (frame_size, frame_size))

print(dataset)

iSbj = 0
idx_frame = 0
f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
label0 = tf.keras.utils.HDF5Matrix(filename, l0_dataset)

## build the network layers
features_input = tf.keras.Input(shape=frame_size) # add 1 channel because it is black or white shape=(None, 52, 58, 1)
print(features_input)

#First we go down the layers
features = tf.keras.layers.Conv2D(32, 7, activation='relu',padding='SAME')(features_input) #32 filters and 7x7 kernel size. Makes it (None,52,58,32) because we have padded
print(features)

features = tf.keras.layers.MaxPool2D(3,padding='SAME')(features) # window (or kernel) that it looks in is 3x3. Features size is now (None, 18, 20, 32) because 1/3 of the size
print(features)

features_block_1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(features) # size (None, 13, 15, 64)
print(features_block_1)

#Then we go back up the layers
features_up_b_1 = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu')(features_block_1) # size (None, 15, 17, 32)
print(features_up_b_1)

features_up_b_2 = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu')(features_up_b_1) # size (None, 13, 15, 64)
print(features_up_b_1)

# #don't bother with shuffling and batches for now
# model.fit(dataset, epochs=int(3))
# print('Training done.')

# plt.show()