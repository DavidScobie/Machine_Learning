import os
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py

filename = './data/dataset70-200.h5'
frame1 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0191_004' ),255)
label1 = tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_00')

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Add the image to a batch
image = tf.expand_dims(frame1, 0)
image = tf.expand_dims(image, 3)

plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")

plt.show()
'''
from tf.keras.preprocessing.image import ImageDataGenerator
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(image, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
model.fit(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
'''