import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

# num_subjects = 180
num_subjects = 60
filename = './data/dataset70-200.h5'
subject_indices = range(num_subjects)

frame_size = np.array([58,52,1])

validation_split = 0.25
num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
num_validation = num_subjects - num_training
training_indices = range(num_training)
validation_indices = range(num_training,num_subjects)
test_indices = range(191,192)

def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        if len(relevant_keys) > 1: #case 64 only has 1 frame
            frame_indic = np.random.randint(0,high=len(relevant_keys)-1)
            f_dataset = 'frame_%04d_%03d' % (iSbj, frame_indic)
            frame = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255),dtype=tf.float32)
            l0_dataset = 'label_%04d_%03d_00' % (iSbj, frame_indic)
            label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
            #Does it contain prostate or not?
            lab = 0
            if tf.math.reduce_max(label0) == 1:
                lab = 1
            yield(tf.expand_dims(frame, axis=2),lab)

def my_test_generator(subject_indices):
    for iSbj in subject_indices:
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        if len(relevant_keys) > 1: #case 64 only has 1 frame
            frame_indic = np.random.randint(0,high=len(relevant_keys)-1)
            test_frame = frame_indic
            print(test_frame)
            f_dataset = 'frame_%04d_%03d' % (iSbj, frame_indic)
            frame = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255),dtype=tf.float32)
            yield(tf.expand_dims(frame, axis=2))

training_dataset = tf.data.Dataset.from_generator(generator = lambda: my_data_generator(subject_indices=training_indices), 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size, ()))

validation_dataset = tf.data.Dataset.from_generator(generator = lambda: my_data_generator(subject_indices=validation_indices), 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size, ()))

test_dataset = tf.data.Dataset.from_generator(generator = lambda: my_test_generator(subject_indices=test_indices), 
                                         output_types = (tf.float32),
                                         output_shapes = (frame_size))


training_batch = training_dataset.shuffle(buffer_size=1024).batch(1)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(1)
test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)
# test_batch = validation_dataset.shuffle(buffer_size=1024).batch(1)

model = VGG16(weights = None, include_top = True, classes=2, input_shape = [58,52,1])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])

history_callback = model.fit(training_batch, epochs=int(5),validation_data = validation_batch)

print('Training done.')

y_pred = model.predict(test_batch)
print(y_pred)
'''
#image test frame
test_frame = 3
t_frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_%03d_00' % (test_frame) ),255)
img = tf.image.convert_image_dtype(t_frame, tf.float32)
plt.imshow(img)
plt.show()
'''

