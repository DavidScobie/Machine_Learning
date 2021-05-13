import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json

p_m_fname = './cluster/pred_masks/T5_rand_frame_p_m.txt'
rand_frame_p_m = np.loadtxt(p_m_fname)
print(np.shape(rand_frame_p_m))

m_v_fname = './cluster/pred_masks/T5_rand_frame_m_v.txt'
rand_frame_m_v = np.loadtxt(m_v_fname)

truth_2 = 2*rand_frame_m_v
overlay = truth_2 + rand_frame_p_m
plt.figure(0)
plt.imshow(overlay)

true_array = np.reshape(rand_frame_m_v,(3016,1))

#I need to use the saved model in order to get the y_pred array

json_file = open('./cluster/jsons/T5_rand_frame.json', 'r') 
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./cluster/weights/T5_rand_frame_weights.h5') 
print("Loaded model from disk")

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

filename = './data/dataset70-200.h5'
frame_size = [58,52,1]

biggun = tf.Variable(tf.zeros([10,58,52], tf.float32))
for i in range (10):
    test_indices = range(189+i,190+i)

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
    print(y_pred)

    biggun[i,:,:].assign(tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32)))


print(biggun)

for i in range (10):
    plt.figure(i+1)
    plt.imshow(biggun[i,:,:])



# pred_array = np.reshape(test_pred,(3016,1))

# mean = (pred_array + true_array)/2 
# diff = true_array - pred_array

# plt.figure(2)
# plt.plot(mean,diff, 'bo')
# plt.xlabel('mean')
# plt.ylabel('difference')

plt.show()