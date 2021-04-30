import os
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random as rd

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

#only want the first 160 subjects to have data augmented
num_subjects = 2
subject_indices = range(num_subjects)

for iSbj in subject_indices:
    relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
    idx_frame_indics = range(len(relevant_keys))
    print(len(relevant_keys))
    for idx_frame in idx_frame_indics:
        f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
        print(f_dataset)