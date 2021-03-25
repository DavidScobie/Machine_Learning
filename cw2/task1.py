import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 

num_subjects = 200
filename = './data/dataset70-200.h5'
subject_indicies = range(num_subjects)
num_frames_per_subject = 1 #In reality it is average of 27 per person
# dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
# frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset)) / 255

frame1 = tf.transpose(tf.keras.utils.HDF5Matrix(filename, 'frame_0004_003' )) / 255
print(frame1)
# print(tf.math.reduce_max(frame1))

#image a slice
img = tf.image.convert_image_dtype(frame1, tf.float32)
print(np.shape(img))
plt.imshow(img)

##MY DATA GENERATOR
def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        idx_frame = 3
        f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
        frame = tf.keras.utils.HDF5Matrix(filename, f_dataset) / 255
        l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
        label1 = tf.keras.utils.HDF5Matrix(filename, l0_dataset)
        l1_dataset = 'label_%04d_%03d_01' % (iSbj, idx_frame)
        label2 = tf.keras.utils.HDF5Matrix(filename, l1_dataset)
        l2_dataset = 'label_%04d_%03d_02' % (iSbj, idx_frame)
        label3 = tf.keras.utils.HDF5Matrix(filename, l2_dataset)
        stacked = tf.stack([frame,label1,label2,label3],axis = 2)
        # print(stacked)
        # yield (tf.expand_dims(frame, axis=2), label)
        yield stacked

subject_indices = range(num_subjects)
# subject_indicies = np.array([4])
training_dataset = my_data_generator(subject_indicies)
print(training_dataset)
frame_size = np.array([52,58])

dataset = tf.data.Dataset.from_generator(generator = my_data_generator, 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))
print(dataset)                                


# plt.show()


# dataset = tf.data.Dataset.from_generator(generator = my_data_generator, 
#                                          output_types = (tf.float32, tf.int32),
#                                          output_shapes = (frame_size+[1], ()))


## DATA GENERATOR FROM CLASSIFICATION (h5 FILES)
# num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
# validation_split = 0.2
# num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
# num_validation = num_subjects - num_training
# training_indices = range(num_training)
# validation_indices = range(num_training,num_subjects)
# num_frames_per_subject = 1
# def data_generator(subject_indices):
#     for iSbj in subject_indices:
#         dataset = '/subject%06d_num_frames' % iSbj
#         num_frames = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
#         idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
#         dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
#         frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset)) / 255
#         dataset = '/subject%06d_label%08d' % (iSbj, idx_frame)
#         label = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
#         yield (tf.expand_dims(frame, axis=2), label)


##SEGMENTATION NPY FILE DATA READER
# path_to_data = './data/dataset70-200'
# path_to_save = './result'

# ## npy data loader class
# class DataReader:
#     def __init__(self, folder_name):
#         self.folder_name = folder_name
#     def load_images_train(self, indices_mb):
#         return self.load_npy_files(["image_train%02d.npy" % idx for idx in indices_mb])
#     def load_images_test(self, indices_mb):
#         return self.load_npy_files(["image_test%02d.npy" % idx for idx in indices_mb])
#     def load_labels_train(self, indices_mb):
#         return self.load_npy_files(["label_train%02d.npy" % idx for idx in indices_mb])
#     def load_npy_files(self, file_names):
#         images = [np.float32(np.load(os.path.join(self.folder_name, fn))) for fn in file_names]
#         return np.expand_dims(np.stack(images, axis=0), axis=4)[:, ::2, ::2, ::2, :]

# DataFeeder = DataReader(path_to_data)

# input_mb = DataFeeder.load_images_train(indices_mb)