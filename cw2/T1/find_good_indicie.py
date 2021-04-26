import os
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

filename = './data/dataset70-200.h5'
# f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
for i in range (15):
    lab = 'label_0191_%03d_00' % (i)
    fra = 'frame_0191_%03d' % (i)

    label = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, lab ),255)
    frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, fra ),255)

    plt.figure(2*i)
    plt.imshow(tf.image.convert_image_dtype(frame, tf.float32))
    plt.figure((2*i)+1)
    plt.imshow(tf.image.convert_image_dtype(label, tf.float32))

plt.show()

