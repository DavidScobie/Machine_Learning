import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

num_subjects = 180
filename = './data/dataset70-200.h5'
subject_indices = range(num_subjects)

frame_size = np.array([58,52,1])
model = VGG16(weights = None, include_top = True, classes=2, input_shape = [52,58,1])
# model = VGG16(input_shape=frame_size,classes=2, include_top = False, weights = None)

print(model.summary())

# model.predict()