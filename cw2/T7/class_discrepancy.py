import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

filename = './data/dataset70-200.h5'

lab = 0
count = 0
for iSbj in range(191,192):
    relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
    idx_frame_indics = range(len(relevant_keys))
    for idx_frame in idx_frame_indics:
        if len(relevant_keys) > 1: #case 64 only has 1 frame
            f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
            frame = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255),dtype=tf.float32)

            l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
            label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
            l1_dataset = 'label_%04d_%03d_01' % (iSbj, idx_frame)
            label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l1_dataset),dtype=tf.float32)
            l2_dataset = 'label_%04d_%03d_02' % (iSbj, idx_frame)
            label2 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l2_dataset),dtype=tf.float32)

            is0 = tf.math.reduce_max(label0)
            is1 = tf.math.reduce_max(label1)
            is2 = tf.math.reduce_max(label2)
            
            #Does it contain prostate or not?
            count = count + 1
            is_it = 0
            if is0+is1+is2 >= 2: #yes loop
                lab = lab + 1
                is_it = 1
            print(is_it)
print(count)
print(lab)           
            