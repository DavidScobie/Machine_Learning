import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import h5py

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

num_subjects = 4
filename = './data/dataset70-200.h5'
subject_indices = range(num_subjects)


#I think this needs to be by 3 (or maybe 4) as we have a stack of 1 frame and 3 labels in the generator 
frame_size = np.array([58,52,1])

#image a slice
frame1 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0000_000_02' ),255)
# frame1 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0000_000' ),255)
img = tf.image.convert_image_dtype(frame1, tf.float32)
plt.imshow(img)
# plt.show()

##MY DATA GENERATOR

validation_split = 0.25
num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
num_validation = num_subjects - num_training
training_indices = range(num_training)
validation_indices = range(num_training,num_subjects)
test_indices = range(50,51)

def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        # idx_frame_indics = range(num_subjects)
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        idx_frame_indics = range(len(relevant_keys))
        for idx_frame in idx_frame_indics:
            f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
            frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
            l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
            label0 = tf.keras.utils.HDF5Matrix(filename, l0_dataset)
            yield(tf.expand_dims(frame, axis=2), tf.expand_dims(label0, axis=2))

def my_test_generator(subject_indices):
    for iSbj in subject_indices:
        # idx_frame_indics = range(num_subjects)
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        # idx_frame_indics = range(len(relevant_keys))
        idx_frame_indics= range(1)
        for idx_frame in idx_frame_indics:
            f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
            frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
            yield(tf.expand_dims(frame, axis=2))

training_dataset = tf.data.Dataset.from_generator(generator = lambda: my_data_generator(subject_indices=training_indices), 
                                         output_types = (tf.float32, tf.float32),
                                         output_shapes = (frame_size, frame_size))

validation_dataset = tf.data.Dataset.from_generator(generator = lambda: my_data_generator(subject_indices=validation_indices), 
                                         output_types = (tf.float32, tf.float32),
                                         output_shapes = (frame_size, frame_size))

test_dataset = tf.data.Dataset.from_generator(generator = lambda: my_test_generator(subject_indices=test_indices), 
                                         output_types = (tf.float32),
                                         output_shapes = (frame_size))


print(training_dataset)
training_batch = training_dataset.shuffle(buffer_size=1024).batch(4)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(4)
test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

## build the network layers
features_input = tf.keras.Input(shape=frame_size) # add 1 channel because it is black or white shape=(None, 52, 58, 1)
print(features_input)

#First we go down the layers
features = tf.keras.layers.Conv2D(32, 7, activation='relu',padding='SAME')(features_input) #32 filters and 7x7 kernel size. Makes it (None,52,58,32) because we have padded
print(features)

#If you don't specify strides it will default to pool size
features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='SAME')(features) # (None,26,29,1) 
print(features)

features_block_1 = tf.keras.layers.Conv2D(64, 3, activation='relu',padding='SAME')(features) # size (None, 26, 29, 64)
print(features_block_1)

# features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_1)
# features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
# features_block_2 = features + features_block_1

# features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_2)
# features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
# features = features + features_block_2

#Then we go back up the layers
features = tf.keras.layers.UpSampling2D(size=(2, 2))(features_block_1) # (None,52,58,1)
print(features)

features_up_b_1 = tf.keras.layers.Conv2DTranspose(32, 3,padding='SAME')(features) # size (None, 52, 58, 32)
print(features_up_b_1)

features_up_b_2 = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid',padding='SAME')(features_up_b_1) # size (None, 52, 58, 32)
print(features_up_b_2)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=features_up_b_2)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            #   loss='sparse_categorical_crossentropy',
              loss = 'MeanSquaredError',
              metrics=['MeanAbsoluteError'])

#don't bother with shuffling and batches for now
history_callback = model.fit(training_batch, epochs=int(3),validation_data = validation_batch)
print('Training done.')

#try the a frame to test the model
# test_data = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0050_000' ),255)
print(test_batch)
y_pred = model.predict(test_batch)
print(tf.shape(y_pred))

test_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
plt.imshow(tf.squeeze(test_pred))
# plt.show()