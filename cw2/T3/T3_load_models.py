import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
# import os
# import tensorflow.keras.losses
# tensorflow.keras.losses.keras_custom_loss_function = keras_custom_loss_function

def keras_custom_loss_function(y_actual,y_predicted):
    eps = 1e-6
    numer = 2*kb.sum(y_predicted*y_actual)
    denom = eps + kb.sum(y_predicted) + kb.sum(y_actual)
    return 1 - kb.mean(numer/denom)

frame_size = np.array([58,52,1])

# modelFile = 'my_model_00.h5'

# model = model_from_json(open(modelFile).read())
# model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))

# model = load_model('my_model_00.h5')

# load json and create model
json_file = open('mod_00.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('weights_00.h5')
print("Loaded model from disk")


f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

filename = './data/dataset70-200.h5'

test_indices = range(191,192)

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

test_dataset = tf.data.Dataset.from_generator(generator = lambda: my_test_generator(subject_indices=test_indices), 
                                        output_types = (tf.float32),
                                        output_shapes = (frame_size))

test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

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

plt.show()
