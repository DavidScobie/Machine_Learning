import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np 
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

f = h5py.File('./data/dataset70-200.h5','r')
keys = f.keys()

num_subjects = 4
filename = './data/dataset70-200.h5'
subject_indices = range(num_subjects)


#I think this needs to be by 3 (or maybe 4) as we have a stack of 1 frame and 3 labels in the generator 
frame_size = np.array([58,52,1])

##MY DATA GENERATOR
validation_split = 0.25
num_training = int(tf.math.floor(num_subjects*(1-validation_split)).numpy())
num_validation = num_subjects - num_training
training_indices = range(num_training)
validation_indices = range(num_training,num_subjects)
test_indices = range(191,192)

#Define augmentation image data generator
datagen=ImageDataGenerator(rotation_range=90,
                        horizontal_flip=True,
                        vertical_flip=True)

#set training batch size
t_b_size = 2

def my_data_generator(subject_indices):
    for iSbj in subject_indices:
        # idx_frame_indics = range(num_subjects)
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]

        if len(relevant_keys) > 1: #case 64 only has 1 frame
            #Instead of for loop through all, just do 1 frame
            frame_indic = np.random.randint(0,high=len(relevant_keys)) 

            f_dataset = 'frame_%04d_%03d' % (iSbj, frame_indic)
            frame = tf.cast(tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255),dtype=tf.float32)
            
            #randomly pick one label from the 3
            label_indic = np.random.randint(0,high=2) 
            lab_dataset = 'label_%04d_%03d_%02d' % (iSbj, frame_indic, label_indic)
            label = tf.cast(tf.keras.utils.HDF5Matrix(filename, lab_dataset),dtype=tf.float32)
            
            #data augmentation
            ran_num = np.random.randint(1,high=100)
            if ran_num <= 20:
                    # Add the image to a batch
                image = tf.expand_dims(frame, 0)
                image = tf.expand_dims(image, 3)
                mask = tf.expand_dims(label, 0)
                mask = tf.expand_dims(mask, 3)

                seed = np.random.randint(10,high=100000)
                imagegen=datagen.flow(image,batch_size=t_b_size,seed=seed)
                imagegen2=datagen.flow(mask,batch_size=t_b_size,seed=seed)

                x=imagegen.next()
                y=imagegen2.next()

                frame = tf.squeeze(tf.convert_to_tensor(x[0]))
                label0 = tf.squeeze(tf.convert_to_tensor(y[0]))

            yield(tf.expand_dims(frame, axis=2), tf.expand_dims(label, axis=2))


def my_test_generator(subject_indices):
    for iSbj in subject_indices:
        # idx_frame_indics = range(num_subjects)
        relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
        # idx_frame_indics = range(len(relevant_keys))
        idx_frame_indics= range(4,6)
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



training_batch = training_dataset.shuffle(buffer_size=1024).batch(t_b_size)
validation_batch = validation_dataset.shuffle(buffer_size=1024).batch(t_b_size)
test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

## build the network layers
features_input = tf.keras.Input(shape=frame_size) # add 1 channel because it is black or white shape=(None, 52, 58, 1)


#Pad with zeros to make nicely divisible shape
features = tf.keras.layers.ZeroPadding2D(padding=(1, 4))(features_input)


#First we go down the layers
features_block_1 = tf.keras.layers.Conv2D(32, 3, activation='relu',padding='SAME')(features) #32 filters and 7x7 kernel size. (None,60,60,32)


#batch normalisation
features_block_1 = tf.keras.layers.BatchNormalization()(features_block_1)

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features_block_1)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_2 = features + features_block_1

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features_block_2)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_3 = features + features_block_2

#If you don't specify strides it will default to pool size
features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='SAME')(features_block_3) 


features_block_4 = tf.keras.layers.Conv2D(64, 3, activation='relu',padding='SAME')(features) #(None,30,30,64) 


features_block_4 = tf.keras.layers.BatchNormalization()(features_block_4)

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_4)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_5 = features + features_block_4

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_5)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_6 = features + features_block_5

#Reducing size again
features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='SAME')(features_block_6) 


features_block_7 = tf.keras.layers.Conv2D(128, 3, activation='relu',padding='SAME')(features) #(None,15,15,128) 


features_block_7 = tf.keras.layers.BatchNormalization()(features_block_7)

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_7)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_8 = features + features_block_7

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_8)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_9 = features + features_block_8

features = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(3, 3),padding='SAME')(features_block_9) 

#Change to 384 filters when it starts to actually work
features_block_10 = tf.keras.layers.Conv2D(384, 3, activation='relu',padding='SAME')(features) #(None,5,5,384) 


features_block_10 = tf.keras.layers.BatchNormalization()(features_block_10)

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
#Upsampling layer 1
features_block_14 = tf.keras.layers.Conv2DTranspose(128, strides=(3,3), kernel_size = (3,3), activation='relu')(features_block_13) + features_block_9 # (None,15,15,128)


features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_14)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_15 = features + features_block_14

features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_15)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_16 = features + features_block_15

#Upsampling layer 2
features_block_17 = tf.keras.layers.Conv2DTranspose(64, strides=(2,2), kernel_size = (2,2), activation='relu')(features_block_16) + features_block_6 # (None,30,30,64)


features_block_17 = tf.keras.layers.BatchNormalization()(features_block_17)

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_17)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_18 = features + features_block_17

features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_18)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_19 = features + features_block_18

#Upsampling layer 3
features_block_20 = tf.keras.layers.Conv2DTranspose(32, strides=(2,2), kernel_size = (2,2), activation='relu')(features_block_19) + features_block_3 # (None,60,60,32)


features_block_20 = tf.keras.layers.BatchNormalization()(features_block_20)

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features_block_20)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_21 = features + features_block_20

features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features_block_21)
features = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(features)
features_block_22 = features + features_block_21

features_up_b_1 = tf.keras.layers.Conv2DTranspose(32, 3,padding='SAME')(features_block_22) # size (None, 52, 58, 32)


features_up_b_2 = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid',padding='SAME')(features_up_b_1) # size (None, 52, 58, 32)

#crop to get correct output shape
features_end = tf.keras.layers.Cropping2D(cropping=((1, 1), (4, 4)))(features_up_b_2)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=features_end)
model.summary()

def keras_custom_loss_function(y_actual,y_predicted):
    eps = 1e-6
    numer = 2*kb.sum(y_predicted*y_actual)
    denom = eps + kb.sum(y_predicted) + kb.sum(y_actual)
    return 1 - kb.mean(numer/denom)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss = keras_custom_loss_function,
            #   loss = 'MeanSquaredError',
            metrics=['MeanAbsoluteError'])

#don't bother with shuffling and batches for now
history_callback = model.fit(training_batch, epochs=int(1),validation_data = validation_batch)

print('Training done.')

#try a frame to test the model
y_pred = model.predict(test_batch)

print(y_pred)
test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
test_pred = tf.transpose(test_pred, perm=[1, 2, 0])
print(test_pred)
# plt.figure(1)
# plt.imshow(test_pred)

#Put a 0.5 threshold on the prediction
test_pred_mask = tf.Variable(tf.zeros([58,52,2], tf.int32))
for ind in range(4,6):
    for i in range (58):
        for j in range (52):
            if test_pred[i][j][ind-4] >= 0.5:
                test_pred_mask[i,j,ind-4].assign(1)

# plt.figure(2) #image the binary mask
# test_pred_mask = tf.image.convert_image_dtype(test_pred_mask, tf.int32)
# plt.imshow(test_pred_mask) 
# np.savetxt('./pred_masks/pred_mask.txt', tf.image.convert_image_dtype(test_pred_mask, tf.int32).numpy())


#Dealing with test data


maj_label = tf.Variable(tf.zeros([58,52,2], tf.int32))
# stck_sum_of_labs = tf.Variable(tf.zeros([58,52,2], tf.int32))

for ind in range(4,6):
    #need to take the consensus label as truth
    l0_dataset = 'label_%04d_%03d_00' % (191, ind)
    l1_dataset = 'label_%04d_%03d_01' % (191, ind)
    l2_dataset = 'label_%04d_%03d_02' % (191, ind)

    label0 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l0_dataset),dtype=tf.float32)
    label1 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l1_dataset),dtype=tf.float32)
    label2 = tf.cast(tf.keras.utils.HDF5Matrix(filename, l2_dataset),dtype=tf.float32)
    print(tf.math.reduce_max(label0))
    print(tf.math.reduce_max(label1))
    print(tf.math.reduce_max(label2))
    sum_of_labs = label0+label1+label2
    
    for i in range (58):
        for j in range (52):
            if sum_of_labs[i][j] >= 2:
                maj_label[i,j,ind-4].assign(1)
    print(maj_label)
    print(tf.math.reduce_max(maj_label))

#Image the corresponding frame and label
# test_frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, 'frame_0191_004' ),255)
# test_frame_img = tf.image.convert_image_dtype(test_frame, tf.float32)
# plt.figure(3)
# plt.imshow(test_frame_img)
maj_label = tf.image.convert_image_dtype(maj_label, tf.int32)
# plt.figure(4)
# plt.imshow(maj_label_img)



# print(maj_label_img)
print(test_pred_mask)

print(maj_label)

#find out if the points are the same on both (for the mask)
match = tf.Variable(tf.zeros([58,52,2], tf.int32))
for ind in range(4,6):
    for i in range (58):
        for j in range (52):
            if maj_label[i][j][ind-4] == test_pred_mask[i][j][ind-4]:
            # if maj_label[i,j,ind-4] == test_pred_mask[i,j,ind-4]:
                match[i,j,ind-4].assign(1)
print(match)
print(tf.math.reduce_sum(match))
print(tf.math.reduce_sum(match)/(58*52*2))

#saving training loss logs
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
loss_fname = './loss/loss_history.txt' 
np.savetxt(loss_fname, numpy_loss_history, delimiter=",")

#saving validation loss logs
val_loss_history = history_callback.history["val_loss"]
numpy_val_loss_history = np.array(val_loss_history)
val_loss_fname = './loss/val_loss_history.txt' 
np.savetxt(val_loss_fname,numpy_val_loss_history, delimiter=",")

# plt.show()
