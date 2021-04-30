import os
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
mask = tf.expand_dims(label1, 0)
mask = tf.expand_dims(mask, 3)
'''
plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")

# plt.show()


# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     horizontal_flip=True,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(image, augment=True, seed=seed)
mask_datagen.fit(mask, augment=True, seed=seed)
# image_generator = image_datagen.flow_from_directory(
#     'data/images',
#     class_mode=None,
#     seed=seed)
# mask_generator = mask_datagen.flow_from_directory(
#     'data/masks',
#     class_mode=None,
#     seed=seed)


# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
# model.fit(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)
'''
datagen=ImageDataGenerator(rotation_range=90,
                           horizontal_flip=True,
                           vertical_flip=True)
seed = 1
imagegen=datagen.flow(image,batch_size=1,seed=seed)
imagegen2=datagen.flow(mask,batch_size=1,seed=seed)
x=imagegen.next()
y=imagegen2.next()
print(np.shape(x))
plt.figure(0)
plt.imshow(x[0].astype('float32'))
# plt.figure(1)
# plt.imshow(x[1][0].astype('float32'))
plt.figure(2)
plt.imshow(y[0].astype('float32'))
# plt.figure(3)
# plt.imshow(y[1][0].astype('float32'))
plt.show()



# datagen = ImageDataGenerator(shear_range=0.02,dim_ordering=K._image_dim_ordering,rotation_range=5,width_shift_range=0.05, height_shift_range=0.05,zoom_range=0.3,fill_mode=’constant’, cval=0)

# for samples in range(0,100):
# seed = rd.randint(low=10,high=100000)
# for imags_batch in datagen.flow(imgs_train,batch_size=batch_size,save_to_dir=’augmented’,save_prefix=’aug’,seed=seed,save_format=’tif’):
# print(‘-‘)
# break
# for imgs_mask_batch in datagen.flow(imgs_mask_train, batch_size=batch_size, save_to_dir=’augmented’,seed=seed, save_prefix=’mask_aug’,save_format=’tif’):
# print(‘|’)
# break
# print((samples+1)*batch_size)