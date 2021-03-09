# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import random

import tensorflow as tf

print('1')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["home"]="0"
filename = './data/ultrasound_50frames.h5'
frame_size = tf.keras.utils.HDF5Matrix(filename, '/frame_size').data.value
frame_size = [frame_size[0][0],frame_size[1][0]]
num_classes = tf.keras.utils.HDF5Matrix(filename, '/num_classes').data.value[0][0]
print(frame_size)
print(num_classes)

## build the network layers
features_input = tf.keras.Input(shape=frame_size+[1])
print(features_input)
features = tf.keras.layers.Conv2D(32, 7, activation='relu')(features_input)

features = tf.keras.layers.MaxPool2D(3)(features)
features_block_1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(features)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_1)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_2 = features + features_block_1
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_2)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features = features + features_block_2

features = tf.keras.layers.MaxPool2D(3)(features)
features_block_3 = tf.keras.layers.Conv2D(128, 3, activation='relu')(features)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_3)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_4 = features + features_block_3
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_4)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_5 = features + features_block_4
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_5)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_6 = features + features_block_5
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_6)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features = features + features_block_6

features = tf.keras.layers.Conv2D(128, 3, activation='relu')(features)
features = tf.keras.layers.GlobalAveragePooling2D()(features)
features = tf.keras.layers.Dense(units=256, activation='relu')(features)
features = tf.keras.layers.Dropout(0.5)(features)
logits_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(features)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=logits_output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])

## data loader using a generator
num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
validation_split = 0.2
num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
num_validation = num_subjects - num_training
# subject_indices = range(num_subjects)
training_indices = range(num_training)
validation_indices = range(num_training,num_subjects)
num_frames_per_subject = 1
# def data_generator():
def data_generator(subject_indices):
    for iSbj in subject_indices:
        dataset = '/subject%06d_num_frames' % iSbj
        num_frames = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
        dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
        frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset)) / 255
        dataset = '/subject%06d_label%08d' % (iSbj, idx_frame)
        label = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        yield (tf.expand_dims(frame, axis=2), label)

# dataset = tf.data.Dataset.from_generator(generator = data_generator, 
#                                          output_types = (tf.float32, tf.int32),
#                                          output_shapes = (frame_size+[1], ()))

training_dataset = tf.data.Dataset.from_generator(generator = lambda: data_generator(subject_indices=training_indices), 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))

validation_dataset = tf.data.Dataset.from_generator(generator = lambda: data_generator(subject_indices=validation_indices), 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))


## training
# dataset_batch = dataset.shuffle(buffer_size=1024).batch(32)
training_batch = training_dataset.shuffle(buffer_size=1024).batch(32)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(32)
# model.fit(dataset_batch, epochs=int(10))
model.fit(training_batch, epochs=int(10),validation_data = validation_batch)
print('Training done.')
