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

#initialise arrays
stck_test_preds = tf.Variable(tf.zeros([10,58,52], tf.float32))

stc_t_p_arr = np.zeros([10,58,52])
pred_array = np.zeros([10,3016,1])
# mean = np.zeros([10,3016,1])
# diff = np.zeros([10,3016,1])
numer = np.zeros([10])
denom = np.zeros([10])
DICE = np.zeros([10])
eps = 1e-6

for i in range (10):
    #first of all getting the true label

        #majority voting
    l0_dataset = 'label_%04d_%03d_00' % (189+i, 4)
    l1_dataset = 'label_%04d_%03d_01' % (189+i, 4)
    l2_dataset = 'label_%04d_%03d_02' % (189+i, 4)

    label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
    label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l1_dataset),dtype=tf.float32)
    label2 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l2_dataset),dtype=tf.float32)

    sum_of_labs = label0+label1+label2

    sum_of_labs_shape = sum_of_labs.shape
    maj_label = tf.Variable(tf.zeros([sum_of_labs_shape[0],sum_of_labs_shape[1]], tf.int32))
    for k in range (sum_of_labs_shape[0]):
        for j in range (sum_of_labs_shape[1]):
            if sum_of_labs[k][j] >= 2:
                maj_label[k,j].assign(1)
    print(tf.convert_to_tensor(maj_label))
    # squeezed = tf.squeeze(tf.image.convert_image_dtype(true_frame, tf.float32))
    numpy = tf.make_ndarray(tf.make_tensor_proto(tf.convert_to_tensor(maj_label)))
    true_array = np.reshape(numpy,(3016,1))


    #now working with model predictions
    test_indices = range(189+i,190+i)
    idx_frame_indics= range(4,5)
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

    stck_test_preds[i,:,:].assign(tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32)))

    # plt.figure(i+1)
    # plt.imshow(stck_test_preds[i,:,:])

    stc_t_p_arr[i,:,:] = (tf.make_ndarray(tf.make_tensor_proto(stck_test_preds[i,:,:])))

    pred_array[i,:,:] = (np.reshape(stc_t_p_arr[i,:,:],(3016,1)))

    numer[i] = 2*np.sum(true_array*pred_array[i,:,:],axis = 0)
    denom[i] = eps + np.sum(pred_array[i,:,:],axis = 0) + np.sum(true_array,axis = 0)
    DICE[i] = 1 - numer[i]/denom[i]
    
    
    # plt.figure(i+11)
    # plt.imshow(numpy)
print(DICE)
acc = 1-DICE
print(acc)
    # print(pred_array[i,:,:])
    # print(true_array)

    # mean[i,:,:] = (pred_array[i,:,:] + true_array)/2
    # diff[i,:,:] = (true_array - pred_array[i,:,:])
    # print(mean)
    # print(diff)

    # #reshape into column vectors
    # mean = np.reshape(mean,[10*3016])


# plt.figure(2)
# plt.plot(mean,diff, 'bo')
# plt.xlabel('mean')
# plt.ylabel('difference')

plt.show()