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
frame_size = np.array([52,58,3])

#image a slice
frame1 = tf.transpose(tf.keras.utils.HDF5Matrix(filename, 'label_0000_000_02' )) / 255
img = tf.image.convert_image_dtype(frame1, tf.float32)
plt.imshow(img)

##MY DATA GENERATOR
def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        # relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        # idx_frame_indics = range(len(relevant_keys))
        idx_frame_indics = range(num_subjects)
        for idx_frame in idx_frame_indics:
            f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
            frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
            l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
            label1 = tf.keras.utils.HDF5Matrix(filename, l0_dataset)
            l1_dataset = 'label_%04d_%03d_01' % (iSbj, idx_frame)
            label2 = tf.keras.utils.HDF5Matrix(filename, l1_dataset)
            l2_dataset = 'label_%04d_%03d_02' % (iSbj, idx_frame)
            label3 = tf.keras.utils.HDF5Matrix(filename, l2_dataset)
            yield((tf.expand_dims(frame, axis=2), label1, label2, label3))

dataset = tf.data.Dataset.from_generator(generator = my_data_generator, 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size, ()))

print(dataset)
plt.show()