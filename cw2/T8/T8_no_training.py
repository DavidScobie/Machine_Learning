import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import model_from_json

#I want to check what the result of model.predict is for classification

#First I want to load the classification model in
json_file = open('./T5_T7_from_cluster/jsons/T7_opti_b135_600.json', 'r') 
loaded_model_json = json_file.read()
json_file.close()
model_T7 = model_from_json(loaded_model_json)
# load weights into new model
model_T7.load_weights('./T5_T7_from_cluster/weights/T7_opti_b135_600_w.h5') 
print("Loaded T7 model from disk")

# Now I want to load the segmentation model in
json_file = open('./T5_T7_from_cluster/jsons/T5_maj_lab.json', 'r') 
loaded_model_json = json_file.read()
json_file.close()
model_T5 = model_from_json(loaded_model_json)
# load weights into new model
model_T5.load_weights('./T5_T7_from_cluster/weights/T5_maj_lab_w.h5') 
print("Loaded T5 model from disk")

#Now we want the test generator
frame_size = [58,52,1]
f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

filename = './data/dataset70-200.h5'

n_frames = 271
test_indices = range(190,200)

#Now I want to see whether the frame is classified as prostate containing

class_thresh_array = np.array([0.5003,0.50035,0.5004,0.50045,0.5005,0.50055,0.5006,0.50065])
# class_thresh_array = np.array([0.5,0.50062,0.500647,0.5007])

for class_thresh in class_thresh_array:

    count = -1
    match = tf.Variable(tf.zeros([58,52,n_frames], tf.int32))

    for iSbj in test_indices:
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        idx_frame_indics = range(len(relevant_keys))
        # idx_frame_indics= range(4,7)
        for idx_frame in idx_frame_indics:

            def my_test_generator():
                f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
                frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
                yield(tf.expand_dims(frame, axis=2))

            test_dataset = tf.data.Dataset.from_generator(generator = lambda: my_test_generator(), 
                                            output_types = (tf.float32),
                                            output_shapes = (frame_size))

            test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

            y_pred_T7 = model_T7.predict(test_batch)
            # print(y_pred_T7)

            if y_pred_T7 >= class_thresh: #This is the paprameter we are tuning
                
                #In here we have only frames with prostates so we look at segmentation prediction
                y_pred = model_T5.predict(test_batch)

                #Now we want to compare prediction with ground truth
                
                #First of all we rearrange test pred dimensions
                test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))

                #Put a 0.5 threshold on the test prediction
                test_pred_mask = tf.Variable(tf.zeros([58,52], tf.int32))

                for i in range (58):
                    for j in range (52):
                        if test_pred[i][j] >= 0.5:
                            test_pred_mask[i,j].assign(1)

                # print('done 1st loop')
                

                #Dealing with the truth labels now (finding majority)
                maj_label = tf.Variable(tf.zeros([58,52], tf.int32))

                #need to take the consensus label as truth
                l0_dataset = 'label_%04d_%03d_00' % (iSbj, idx_frame)
                l1_dataset = 'label_%04d_%03d_01' % (iSbj, idx_frame)
                l2_dataset = 'label_%04d_%03d_02' % (iSbj, idx_frame)

                label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
                label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l1_dataset),dtype=tf.float32)
                label2 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l2_dataset),dtype=tf.float32)

                sum_of_labs = label0+label1+label2

                for i in range (58):
                    for j in range (52):
                        if sum_of_labs[i][j] >= 2:
                            maj_label[i,j].assign(1)

                # print('done 2nd loop')

                maj_label = tf.image.convert_image_dtype(maj_label, tf.int32)

                #find out if the points are the same on both the mask and majority

                count = count + 1
                
                for i in range (58):
                    for j in range (52):
                        if maj_label[i][j] == test_pred_mask[i][j]:
                        
                            match[i,j,count].assign(1)

                # print(count)

    #Sum the match matrix to find the accuracy 
    print('Accuracy: ',tf.math.reduce_sum(match)/(58*52*(count+1)))

    #Print how many frames got accuracy found
    print('how many frames',count)



