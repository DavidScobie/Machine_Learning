# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import random
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

RESULT_PATH = './result'

os.environ["CUDA_VISIBLE_DEVICES"]="0"
filename = './data/ultrasound_50frames.h5'

# f = h5py.File('./data/ultrasound_50frames.h5')
# keys = f.keys()
# print(keys)

frame_size = tf.keras.utils.HDF5Matrix(filename, '/frame_size').data.value
frame_size = [frame_size[0][0],frame_size[1][0]]
num_classes = tf.keras.utils.HDF5Matrix(filename, '/num_classes').data.value[0][0]
print(frame_size+[1])
print(num_classes)

## build the network layers
features_input = tf.keras.Input(shape=frame_size+[1]) # add 1 channel because it is black or white shape=(None, 92, 128, 1)
print(features_input)
features = tf.keras.layers.Conv2D(32, 7, activation='relu')(features_input) #32 filters and 7x7 kernel size. Makes it (None, 86,122,32) because we haven't padded
print(features)

features = tf.keras.layers.MaxPool2D(3)(features) # window (or kernel) that it looks in is 3x3. Features size is now (None, 28, 40, 32) because 1/3 of the size
print(features)

features_block_1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(features) # size (None, 26, 38, 64)
print(features_block_1)

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_1)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features) # size (None, 26, 38, 64)
features_block_2 = features + features_block_1 #Adding layers to make the architecture

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_2)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features = features + features_block_2

features = tf.keras.layers.MaxPool2D(3)(features) # shape=(None, 8, 12, 64)
print(features)

features_block_3 = tf.keras.layers.Conv2D(128, 3, activation='relu')(features) # shape=(None, 6, 10, 128)
print(features_block_3)

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_3)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_4 = features + features_block_3 # This group of 3 lines is a resnet block. Analagous to resnet_block() in segmentation tutorial
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_4)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_5 = features + features_block_4
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_5)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_6 = features + features_block_5
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_6)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features = features + features_block_6 # shape=(None, 6, 10, 128)

features = tf.keras.layers.Conv2D(128, 3, activation='relu')(features) # shape=(None, 4, 8, 128)
print(features)

features = tf.keras.layers.GlobalAveragePooling2D()(features) # shape=(None, 128) because batch size = 1 and 128 channels
print(features)

features = tf.keras.layers.Dense(units=256, activation='relu')(features) # shape=(None, 256)
print(features)

features = tf.keras.layers.Dropout(0.5)(features) # shape=(None, 256)
print(features)

logits_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(features) # (None, 4)
print(logits_output)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=logits_output)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])


## data loader using a generator
num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
subject_indices = range(num_subjects)
num_frames_per_subject = 1
def data_generator():
    for iSbj in subject_indices:
        dataset = '/subject%06d_num_frames' % iSbj
        num_frames = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
        dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
        frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset)) / 255
        dataset = '/subject%06d_label%08d' % (iSbj, idx_frame)
        label = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        yield (tf.expand_dims(frame, axis=2), label)

dataset = tf.data.Dataset.from_generator(generator = data_generator, 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))
print(dataset)

#image a slice
frame1 = tf.transpose(tf.keras.utils.HDF5Matrix(filename, 'subject000004_frame00000000' )) / 255
img = tf.image.convert_image_dtype(frame1, tf.float32)
plt.imshow(img)
# plt.show()

## training
dataset_batch = dataset.shuffle(buffer_size=1024).batch(32)
# dataset_batch = dataset.shuffle(buffer_size=376832).batch(32)
print(dataset_batch)
model.fit(dataset_batch, epochs=int(1))
print('Training done.')

## save trained model
# model.save(os.path.join(RESULT_PATH,'saved_model_tf'))  # https://www.tensorflow.org/guide/keras/save_and_serialize
# print('Model saved.')


