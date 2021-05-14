
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random as rd

filename = './data/dataset70-200.h5'
frame1 = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0191_004' ),255), dtype=tf.float32)
label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_00'),dtype=tf.float32)
print(frame1)

# Add the image to a batch
image = tf.expand_dims(frame1, 0)
image = tf.expand_dims(image, 3)
mask = tf.expand_dims(label1, 0)
mask = tf.expand_dims(mask, 3)

datagen=ImageDataGenerator(rotation_range=90,
                           horizontal_flip=True,
                           vertical_flip=True)

seed = rd.randint(10,100000)
imagegen=datagen.flow(image,batch_size=1,seed=seed)
imagegen2=datagen.flow(mask,batch_size=1,seed=seed)
print(imagegen)
x=imagegen.next()
y=imagegen2.next()
print(type(x))
print(np.shape(x))

plt.figure(0)
plt.imshow(x[0].astype('float32'))
plt.figure(1)
plt.imshow(y[0].astype('float32'))
print(type(x[0]))
print(np.shape(x[0]))

x0_tensor = tf.convert_to_tensor(x[0])
print(x0_tensor)

x0_tens_flat = tf.squeeze(x0_tensor)
print(x0_tens_flat)

plt.show()



