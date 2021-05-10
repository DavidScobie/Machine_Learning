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

validation_split = 0.25
num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
num_validation = num_subjects - num_training
training_indices = range(num_training)
validation_indices = range(num_training,num_subjects)
test_indices = range(191,192)

t_b_size = 20

def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        if len(relevant_keys) > 1: #case 64 only has 1 frame
            frame_indic = np.random.randint(0,high=len(relevant_keys)-1)
            f_dataset = 'frame_%04d_%03d' % (iSbj, frame_indic)
            frame = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255),dtype=tf.float32)

            l0_dataset = 'label_%04d_%03d_00' % (iSbj, frame_indic)
            label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
            l1_dataset = 'label_%04d_%03d_01' % (iSbj, frame_indic)
            label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l1_dataset),dtype=tf.float32)
            l2_dataset = 'label_%04d_%03d_02' % (iSbj, frame_indic)
            label2 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l2_dataset),dtype=tf.float32)

            is0 = tf.math.reduce_max(label0)
            is1 = tf.math.reduce_max(label1)
            is2 = tf.math.reduce_max(label2)
            
            #Does it contain prostate or not?
            lab = 0
            if is0+is1+is2 >= 2:
                lab = 1
            yield(tf.expand_dims(frame, axis=2),lab)

def my_test_generator(subject_indices):
    for iSbj in subject_indices:
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        if len(relevant_keys) > 1: #case 64 only has 1 frame
            all_frame_indics = len(relevant_keys)-1
            for frame_indic in range(all_frame_indics):
                f_dataset = 'frame_%04d_%03d' % (iSbj, frame_indic)
                print(f_dataset)
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


training_batch = training_dataset.shuffle(buffer_size=1024).batch(t_b_size)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(t_b_size)
test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)
# test_batch = validation_dataset.shuffle(buffer_size=1024).batch(1)

VGG_model = VGG16(weights = None, include_top = False, input_shape = [58,52,1])

model = tf.keras.Sequential()

for layer in VGG_model.layers[:-1]:
    model.add(layer)

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

print(model.summary())

# def keras_custom_loss_function(y_actual,y_predicted):
#     eps = 1e-6
#     numer = 2*kb.sum(y_predicted*y_actual)
#     denom = eps + kb.sum(y_predicted) + kb.sum(y_actual)
#     return 1 - kb.mean(numer/denom)

def keras_custom_loss_function(y_actual,y_predicted):
    y_predicted = tf.cast(y_predicted,tf.float32)
    y_actual = tf.cast(y_actual,tf.float32)
    first_term = y_actual*(kb.log(y_predicted))
    second_term = (1-y_actual)*(kb.log(1-y_predicted))
    return -kb.sum(first_term+second_term)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            #   loss='sparse_categorical_crossentropy',
            #   metrics=['SparseCategoricalAccuracy'])
              loss=keras_custom_loss_function,
              metrics=['binary_crossentropy'])
              

history_callback = model.fit(training_batch, epochs=int(1),validation_data = validation_batch)

print('Training done.')

y_pred = model.predict(test_batch)
print(y_pred)

#saving training loss logs
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt('./loss/loss_history.txt', numpy_loss_history, delimiter=",")

#saving validation loss logs
val_loss_history = history_callback.history["val_loss"]
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt('./loss/val_loss_history.txt',numpy_val_loss_history, delimiter=",")

#saving predictions
np.savetxt('./class_preds/class_pred.txt',y_pred, delimiter=",")

'''
#image test frame
test_frame = 3
t_frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'label_0191_%03d_00' % (test_frame) ),255)
img = tf.image.convert_image_dtype(t_frame, tf.float32)
plt.imshow(img)
plt.show()
'''

