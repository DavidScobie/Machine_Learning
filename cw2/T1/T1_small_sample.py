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
frame1 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_00' ),255)
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
test_indices = range(191,192)

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
        idx_frame_indics= range(4,5)
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
training_batch = training_dataset.shuffle(buffer_size=1024).batch(1)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(1)
test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

## build the network layers
features_input = tf.keras.Input(shape=frame_size) # add 1 channel because it is black or white shape=(None, 52, 58, 1)
print(features_input)

#Pad with zeros to make nicely divisible shape
features = tf.keras.layers.ZeroPadding2D(padding=(1, 4))(features_input)
print(features)

#First we go down the layers
features_block_1 = tf.keras.layers.Conv2D(32, 3, activation='relu',padding='SAME')(features) #32 filters and 7x7 kernel size. (None,60,60,32)
print(features)

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_2 = features + features_block_1

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features_block_2)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_3 = features + features_block_2

#If you don't specify strides it will default to pool size
features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='SAME')(features_block_3) 
print(features)

features_block_4 = tf.keras.layers.Conv2D(64, 3, activation='relu',padding='SAME')(features) #(None,30,30,64) 
print(features_block_1)

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_4)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_5 = features + features_block_4

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_5)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_6 = features + features_block_5

#Reducing size again
features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='SAME')(features_block_6) 
print(features)

features_block_7 = tf.keras.layers.Conv2D(128, 3, activation='relu',padding='SAME')(features) #(None,15,15,128) 
print(features_block_1)

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_7)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_8 = features + features_block_7

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_8)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_9 = features + features_block_8

features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(3, 3),padding='SAME')(features_block_9) 
print(features)
#Change to 384 filters when it starts to actually work
features_block_10 = tf.keras.layers.Conv2D(384, 3, activation='relu',padding='SAME')(features) #(None,5,5,384) 
print(features_block_1)

features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features_block_10)
features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features)
features_block_11 = features + features_block_10

features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features_block_11)
features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features)
features_block_12 = features + features_block_11

features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features_block_12)
features = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')(features)
features_block_13 = features + features_block_12

#Then we go back up the layers
features_block_14 = tf.keras.layers.UpSampling2D(size=(3, 3))(features_block_13) # (None,15,15,128)
print(features)

# features = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(features_block_14)
# features = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(features)
# features_block_15 = features + features_block_14

# features = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(features_block_15)
# features = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(features)
# features_block_16 = features + features_block_15

features_block_17 = tf.keras.layers.UpSampling2D(size=(2, 2))(features_block_14) # (None,30,30,64)
print(features)

# features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_17)
# features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
# features_block_18 = features + features_block_17

# features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_18)
# features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
# features_block_19 = features + features_block_18

features = tf.keras.layers.UpSampling2D(size=(2, 2))(features_block_17) # (None,60,60,32)
print(features)

features_up_b_1 = tf.keras.layers.Conv2DTranspose(32, 3,padding='SAME')(features) # size (None, 52, 58, 32)
print(features_up_b_1)

features_up_b_2 = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid',padding='SAME')(features_up_b_1) # size (None, 52, 58, 32)
print(features_up_b_2)

#crop to get correct output shape
features_end = tf.keras.layers.Cropping2D(cropping=((1, 1), (4, 4)))(features_up_b_2)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=features_end)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            #   loss='sparse_categorical_crossentropy',
              loss = 'MeanSquaredError',
              metrics=['MeanAbsoluteError'])

#don't bother with shuffling and batches for now
history_callback = model.fit(training_batch, epochs=int(1),validation_data = validation_batch)
print('Training done.')

#try a frame to test the model
# test_data = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0050_000' ),255)
print(test_batch)
y_pred = model.predict(test_batch)
print(tf.shape(y_pred))

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
plt.figure(1)
plt.imshow(test_pred)

#Put a 0.5 threshold on the prediction
test_pred_shape = test_pred.shape
test_pred_mask = tf.Variable(tf.zeros([test_pred_shape[0],test_pred_shape[1]], tf.int32))
for i in range (test_pred_shape[0]):
    for j in range (test_pred_shape[1]):
        if test_pred[i][j] >= 0.5:
            test_pred_mask[i,j].assign(1)

plt.figure(2)
plt.imshow(tf.image.convert_image_dtype(test_pred_mask, tf.int32))
np.savetxt(os.path.join('pred_masks', 'pred_mask.txt'), tf.image.convert_image_dtype(test_pred_mask, tf.int32).numpy())

#Image the corresponding frame and label
test_label_0 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_00' ),255)
test_label_1 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_01' ),255)
test_label_2 = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_004_02' ),255)
test_frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0191_004' ),255)
test_label_0_img = tf.image.convert_image_dtype(test_label_0, tf.float32)
test_frame_img = tf.image.convert_image_dtype(test_frame, tf.float32)
test_label_1_img = tf.image.convert_image_dtype(test_label_1, tf.float32)
test_label_2_img = tf.image.convert_image_dtype(test_label_2, tf.float32)
plt.figure(3)
plt.imshow(test_label_0_img)
plt.figure(4)
plt.imshow(test_label_1_img)
plt.figure(5)
plt.imshow(test_label_2_img)
plt.figure(6)
plt.imshow(test_frame_img)

#saving training loss logs
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt(os.path.join('loss', 'loss_history.txt'), numpy_loss_history, delimiter=",")

#saving validation loss logs
val_loss_history = history_callback.history["val_loss"]
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt(os.path.join('loss', 'val_loss_history.txt'), numpy_val_loss_history, delimiter=",")

plt.show()