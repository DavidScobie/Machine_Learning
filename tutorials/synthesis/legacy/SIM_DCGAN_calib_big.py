import tensorflow as tf
import h5py
import numpy as np
import os
# import sys
import random
import time
import tensorflow_helpers as tf_helpers
import tensorflow_tools as tf_tools

flag_randomHyperParam = False

data_set_name = 'im1'
cond_set_name = 'gr1'
filename_data, epoch_size, data_size = tf_helpers.dataset_switcher(data_set_name)
filename_cond, epoch_size_cond, cond_size = tf_helpers.dataset_switcher(cond_set_name)
filename_groups, _, _ = tf_helpers.dataset_switcher('roi')
filename_output_info = 'output_info.txt'
dir_output = os.path.join(os.environ['HOME'], 'Scratch/output/simus0/', "%f" % time.time())
flag_dir_overwrite = os.path.exists(dir_output)
os.makedirs(dir_output)
fid_output_info = open(os.path.join(dir_output, filename_output_info), 'a')

# data set
idx_ROI = 0
idx_crossV = 0
devSet_size = 256

# training
miniBatch_size = 36
totalIterations = 100001
learning_rate = 2e-4
w_std = 0.02
interpMethod = 0
useConditionSmoother = False  # disabled

# model
noise_size = 100
conv_kernel_size = 3
conv_kernel_size_initial = 5
num_channel_initial_G = 512
num_channel_initial_D = 32
lambda_weight_G = 0
lambda_weight_D = 0
lambda_supervised = 0
order_supervised = 2
md_num_features = 0  # to use entire conv layer, set to -1
md_num_kernels = 100
md_kernel_dim = 5
keep_prob_rate = 1
initial_bias = 0.001
useIdentityMapping = True  # disabled
generator_shortcuts = [True, True, True, True, False]

# --- experimental sampling --- # BEFORE seeding!
if flag_randomHyperParam:
    miniBatch_size = int(np.random.choice([16, 25, 36, 49, 64, 100]))
    # devCase_size = num_case-1
    # noise_size = int(np.random.choice([1e2, 500]))
    # idx_ROI = int(np.random.choice([0, 1, 2, 3, 4]))
    idx_crossV = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    num_channel_initial_G = int(np.random.choice([128, 256, 512, 1024]))
    num_channel_initial_D = int(np.random.choice([8, 16, 32, 64]))
    lambda_weight_G = np.random.choice([0, 1e-3, 1e-5])
    lambda_weight_D = np.random.choice([0, 1e-3, 1e-5])
    conv_kernel_size_initial = int(np.random.choice([3, 5, 7]))
    lambda_supervised = np.random.choice([0, 1e-2, 1e-3, 1e-4, 1e-5])
    order_supervised = np.random.choice([1, 2])
    # md_num_features = int(np.random.choice([100, 200, 300, 400, 500]))
    # useConditionSmoother = int(np.random.choice([True, False]))
    learning_rate = np.random.choice([1e-4, 2e-4, 5e-4])
# --- experimental sampling ---


# cross-validation
seed = 38
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)
totalDataIndices = tf_helpers.dataset_indices('tot', idx=idx_ROI, index_file=filename_groups)  # [i for i in range(epoch_size)]
random.shuffle(totalDataIndices)
devDataIndices = tf_helpers.dataset_indices('dev', idx=idx_crossV, data_size=devSet_size, data_indices=totalDataIndices)
trainDataIndices = list(set(totalDataIndices) - set(devDataIndices))
random.shuffle(trainDataIndices)
num_miniBatch = int(len(trainDataIndices)/miniBatch_size)

# debug data saving
log_num_shuffle = 0
log_start_debug = 1000
log_freq_debug = 100
num_repeat_debug = 10  # temp
num_repeat_dev = int(devSet_size/miniBatch_size)
num_repeat_linspace = 30  # temp
noiseDimIndices = [i for i in range(noise_size)]

# TODO: model saving


# information
print('- Algorithm Summary (normalised calibration) --------', flush=True, file=fid_output_info)

print('current_time: %s' % time.asctime(time.gmtime()), flush=True, file=fid_output_info)
print('flag_dir_overwrite: %s' % flag_dir_overwrite, flush=True, file=fid_output_info)
print('idx_ROI: %s' % idx_ROI, flush=True, file=fid_output_info)
print('idx_crossV: %s' % idx_crossV, flush=True, file=fid_output_info)
print('data_set_name: %s' % data_set_name, flush=True, file=fid_output_info)
print('cond_set_name: %s' % cond_set_name, flush=True, file=fid_output_info)
print('miniBatch_size: %s' % miniBatch_size, flush=True, file=fid_output_info)
print('noise_size: %s' % noise_size, flush=True, file=fid_output_info)
print('conv_kernel_size_initial: %s' % conv_kernel_size_initial, flush=True, file=fid_output_info)
print('num_channel_initial_G: %s' % num_channel_initial_G, flush=True, file=fid_output_info)
print('num_channel_initial_D: %s' % num_channel_initial_D, flush=True, file=fid_output_info)
print('lambda_weight_G: %s' % lambda_weight_G, flush=True, file=fid_output_info)
print('lambda_weight_D: %s' % lambda_weight_D, flush=True, file=fid_output_info)
print('lambda_supervised: %s' % lambda_supervised, flush=True, file=fid_output_info)
print('order_supervised: %s' % order_supervised, flush=True, file=fid_output_info)
print('learning_rate: %s' % learning_rate, flush=True, file=fid_output_info)
print('md_num_features: %s' % md_num_features, flush=True, file=fid_output_info)
print('md_num_kernels: %s' % md_num_kernels, flush=True, file=fid_output_info)
print('md_kernel_dim: %s' % md_kernel_dim, flush=True, file=fid_output_info)
print('keep_prob_rate: %s' % keep_prob_rate, flush=True, file=fid_output_info)
print('generator_shortcuts: %s' % generator_shortcuts, flush=True, file=fid_output_info)
print('useIdentityMapping: %s' % useIdentityMapping, flush=True, file=fid_output_info)
print('useConditionSmoother: %s' % useConditionSmoother, flush=True, file=fid_output_info)

print('- End of Algorithm Summary --------', flush=True, file=fid_output_info)
if log_num_shuffle:
    print('trainDataIndices: %s' % trainDataIndices, flush=True, file=fid_output_info)


data_feeder = tf_helpers.TrackedFrameFeeder(filename_data, filename_cond, random_cond=useConditionSmoother, is_grid=True)
noise_feeder = tf_helpers.NoiseFeeder(noise_size, miniBatch_size)

strides_none = [1, 1, 1, 1]
strides_down = [1, 2, 2, 1]
k_conv = [conv_kernel_size, conv_kernel_size]
k_conv0 = [conv_kernel_size_initial, conv_kernel_size_initial]

# basic pre-computes
data_size_1 = [round(i/2) for i in data_size]
data_size_2 = [round(i/2) for i in data_size_1]
data_size_3 = [round(i/2) for i in data_size_2]
data_size_4 = [round(i/2) for i in data_size_3]
data_size_5 = [round(i/2) for i in data_size_4]


# ---------- Generator ----------
g_nc_1p = num_channel_initial_G
g_no_0 = np.prod(data_size_4)*g_nc_1p
G_W1p = tf.get_variable("G_W1p", shape=[noise_size, g_no_0], initializer=tf.random_normal_initializer(0, w_std))
G_b1p = tf.Variable(tf.constant(initial_bias, shape=[g_no_0]), name='G_b1p')
g_nc_1 = g_nc_1p + 3*generator_shortcuts[0]
G_W1 = tf.get_variable("G_W1", shape=k_conv+[g_nc_1, g_nc_1], initializer=tf.random_normal_initializer(0, w_std))
g_nc_2t = round(g_nc_1/2)
G_W2t = tf.get_variable("G_W2t", shape=k_conv+[g_nc_2t, g_nc_1], initializer=tf.random_normal_initializer(0, w_std))
g_nc_2 = g_nc_2t + 3*generator_shortcuts[1]
G_W2 = tf.get_variable("G_W2", shape=k_conv+[g_nc_2, g_nc_2], initializer=tf.random_normal_initializer(0, w_std))
g_nc_3t = round(g_nc_2/2)
G_W3t = tf.get_variable("G_W3t", shape=k_conv+[g_nc_3t, g_nc_2], initializer=tf.random_normal_initializer(0, w_std))
g_nc_3 = g_nc_3t + 3*generator_shortcuts[2]
G_W3 = tf.get_variable("G_W3", shape=k_conv+[g_nc_3, g_nc_3], initializer=tf.random_normal_initializer(0, w_std))
g_nc_4t = round(g_nc_3/2)
G_W4t = tf.get_variable("G_W4t", shape=k_conv+[g_nc_4t, g_nc_3], initializer=tf.random_normal_initializer(0, w_std))
g_nc_4 = g_nc_4t + 3*generator_shortcuts[3]
G_W4 = tf.get_variable("G_W4", shape=k_conv+[g_nc_4, g_nc_4], initializer=tf.random_normal_initializer(0, w_std))
g_nc_5t = round(g_nc_4/2)
G_W5t = tf.get_variable("G_W5t", shape=k_conv+[g_nc_5t, g_nc_4], initializer=tf.random_normal_initializer(0, w_std))
g_nc_5 = g_nc_5t + 3*generator_shortcuts[4]
G_W5 = tf.get_variable("G_W5", shape=k_conv+[g_nc_5, g_nc_5], initializer=tf.random_normal_initializer(0, w_std))
g_nc_o = 1
G_Wo = tf.get_variable("G_Wo", shape=k_conv+[g_nc_5, g_nc_o], initializer=tf.random_normal_initializer(0, w_std))
G_bo = tf.Variable(tf.constant(initial_bias, shape=[g_nc_o]), name='G_bo')

theta_G = [G_W1p, G_b1p, G_W1, G_W2t, G_W2, G_W3t, G_W3, G_W4t, G_W4, G_W5t, G_W5, G_Wo, G_bo]
if lambda_weight_G > 0:
    for i in range(len(theta_G)):
        tf.add_to_collection('g_losses', tf.nn.l2_loss(theta_G[i]) * lambda_weight_G)


def generator(z, y):  # conditioned dc-generator
    g_h1p = tf.reshape(tf.nn.dropout(tf.nn.relu(tf.matmul(z, G_W1p) + G_b1p), keep_prob_ph), [miniBatch_size]+data_size_4+[g_nc_1p])
    if generator_shortcuts[0]:
        g_h1p = tf.concat(3, [g_h1p, tf.image.resize_bilinear(y, data_size_4)])
    g_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(g_h1p, G_W1, strides_none, "SAME")))
    g_h2t = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(g_h1, G_W2t, [miniBatch_size]+data_size_3+[g_nc_2t], strides_down, "SAME")))
    if generator_shortcuts[1]:
        g_h2t = tf.concat(3, [g_h2t, tf.image.resize_bilinear(y, data_size_3)])
    g_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(g_h2t, G_W2, strides_none, "SAME")))
    g_h3t = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(g_h2, G_W3t, [miniBatch_size]+data_size_2+[g_nc_3t], strides_down, "SAME")))
    if generator_shortcuts[2]:
        g_h3t = tf.concat(3, [g_h3t, tf.image.resize_bilinear(y, data_size_2)])
    g_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(g_h3t, G_W3, strides_none, "SAME")))
    g_h4t = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(g_h3, G_W4t, [miniBatch_size]+data_size_1+[g_nc_4t], strides_down, "SAME")))
    if generator_shortcuts[3]:
        g_h4t = tf.concat(3, [g_h4t, tf.image.resize_bilinear(y, data_size_1)])
    g_h4 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(g_h4t, G_W4, strides_none, "SAME")))
    g_h5t = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(g_h4, G_W5t, [miniBatch_size]+data_size + [g_nc_5t], strides_down, "SAME")))
    if generator_shortcuts[4]:
        g_h5t = tf.concat(3, [g_h5t, y])
    g_h5 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(g_h5t, G_W5, strides_none, "SAME")))
    x_sample = tf.nn.dropout(tf.nn.tanh(tf.nn.conv2d(g_h5, G_Wo, strides_none, "SAME") + G_bo), keep_prob_ph)  # NB. no batch_norm for true mean and scale
    return x_sample
# ---------- End of Generator ----------


# ---------- Discriminator ----------
d_nc_1 = num_channel_initial_D
D_W1s = tf.get_variable("D_W1s", shape=k_conv0+[4, d_nc_1], initializer=tf.random_normal_initializer(0, w_std))
D_b1s = tf.Variable(tf.constant(initial_bias, shape=[d_nc_1]), name='D_b1s')
D_W1r1 = tf.get_variable("D_W1r1", shape=k_conv+[d_nc_1, d_nc_1], initializer=tf.random_normal_initializer(0, w_std))
D_W1r2 = tf.get_variable("D_W1r2", shape=k_conv+[d_nc_1, d_nc_1], initializer=tf.random_normal_initializer(0, w_std))

d_nc_2 = d_nc_1*2
D_W2s = tf.get_variable("D_W2s", shape=k_conv+[d_nc_1, d_nc_2], initializer=tf.random_normal_initializer(0, w_std))
D_W2r1 = tf.get_variable("D_W2r1", shape=k_conv+[d_nc_2, d_nc_2], initializer=tf.random_normal_initializer(0, w_std))
D_W2r2 = tf.get_variable("D_W2r2", shape=k_conv+[d_nc_2, d_nc_2], initializer=tf.random_normal_initializer(0, w_std))

d_nc_3 = d_nc_2*2
D_W3s = tf.get_variable("D_W3s", shape=k_conv+[d_nc_2, d_nc_3], initializer=tf.random_normal_initializer(0, w_std))
D_W3r1 = tf.get_variable("D_W3r1", shape=k_conv+[d_nc_3, d_nc_3], initializer=tf.random_normal_initializer(0, w_std))
D_W3r2 = tf.get_variable("D_W3r2", shape=k_conv+[d_nc_3, d_nc_3], initializer=tf.random_normal_initializer(0, w_std))

d_nc_4 = d_nc_3*2
D_W4s = tf.get_variable("D_W4s", shape=k_conv+[d_nc_3, d_nc_4], initializer=tf.random_normal_initializer(0, w_std))
D_W4r1 = tf.get_variable("D_W4r1", shape=k_conv+[d_nc_4, d_nc_4], initializer=tf.random_normal_initializer(0, w_std))
D_W4r2 = tf.get_variable("D_W4r2", shape=k_conv+[d_nc_4, d_nc_4], initializer=tf.random_normal_initializer(0, w_std))

d_nc_5 = d_nc_4*2
D_W5s = tf.get_variable("D_W5s", shape=k_conv+[d_nc_4, d_nc_5], initializer=tf.random_normal_initializer(0, w_std))
D_W5r1 = tf.get_variable("D_W5r1", shape=k_conv+[d_nc_5, d_nc_5], initializer=tf.random_normal_initializer(0, w_std))
D_W5r2 = tf.get_variable("D_W5r2", shape=k_conv+[d_nc_5, d_nc_5], initializer=tf.random_normal_initializer(0, w_std))

d_nc_6 = d_nc_5*2
D_W6s = tf.get_variable("D_W6s", shape=k_conv+[d_nc_5, d_nc_6], initializer=tf.random_normal_initializer(0, w_std))
D_W6r1 = tf.get_variable("D_W6r1", shape=k_conv+[d_nc_6, d_nc_6], initializer=tf.random_normal_initializer(0, w_std))
D_W6r2 = tf.get_variable("D_W6r2", shape=k_conv+[d_nc_6, d_nc_6], initializer=tf.random_normal_initializer(0, w_std))

d_nc_o = 1
d_nf_o = np.prod(data_size_5)*d_nc_6
if md_num_features > 0:  # projected minibatch features
    D_Wmf = tf.get_variable("D_Wmf", shape=[d_nf_o, md_num_features], initializer=tf.random_normal_initializer(0, w_std))
    D_bmf = tf.Variable(tf.constant(initial_bias, shape=[md_num_features]), name='D_bmf')
    D_WmT = tf.get_variable("D_WmT", shape=[md_num_features, md_num_kernels*md_kernel_dim], initializer=tf.random_normal_initializer(0, w_std))
    D_bmT = tf.Variable(tf.constant(initial_bias, shape=[md_num_kernels*md_kernel_dim]), name='D_bmT')
    D_Wo = tf.get_variable("D_Wo", shape=[md_num_features+md_num_kernels, d_nc_o], initializer=tf.random_normal_initializer(0, w_std))
elif md_num_features < 0:  # use entire conv layer
    D_WmT = tf.get_variable("D_WmT", shape=[d_nf_o, md_num_kernels*md_kernel_dim], initializer=tf.random_normal_initializer(0, w_std))
    D_bmT = tf.Variable(tf.constant(initial_bias, shape=[md_num_kernels*md_kernel_dim]), name='D_bmT')
    D_Wo = tf.get_variable("D_Wo", shape=[d_nf_o+md_num_kernels, d_nc_o], initializer=tf.random_normal_initializer(0, w_std))
else:
    D_Wo = tf.get_variable("D_Wo", shape=[d_nf_o, d_nc_o], initializer=tf.random_normal_initializer(0, w_std))
D_bo = tf.Variable(tf.constant(initial_bias, shape=[d_nc_o]), name='D_bo')

theta_D = [D_W1s, D_b1s, D_W1r1, D_W1r2, D_W2s, D_W2r1, D_W2r2, D_W3s, D_W3r1, D_W3r2,  D_W4s, D_W4r1, D_W4r2,
           D_W5s, D_W5r1, D_W5r2, D_W6s, D_W6r1, D_W6r2, D_Wo, D_bo]
if md_num_features > 0:
    theta_D += [D_Wmf, D_bmf, D_WmT, D_bmT]
elif md_num_features < 0:
    theta_D += [D_WmT, D_bmT]
if lambda_weight_D > 0:
    for i in range(len(theta_D)):
        tf.add_to_collection('d_losses', tf.nn.l2_loss(theta_D[i]) * lambda_weight_D)


def discriminator(x, y):  # x = tf.placeholder(tf.float32, [miniBatch_size]+data_size+[1])
    # initial resnet
    d_h1s = tf_tools.nn_leaky_relu(tf.nn.conv2d(tf.concat(3, [x, y]), D_W1s, strides_none, "SAME") + D_b1s)  # NB. no batch_norm for true mean and scale
    d_h1r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h1s, D_W1r1, strides_none, "SAME")))
    d_h1 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h1r, D_W1r2, strides_none, "SAME")) + d_h1s)
    d_h2s = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h1, D_W2s, strides_down, "SAME")))
    d_h2r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h2s, D_W2r1, strides_none, "SAME")))
    d_h2 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h2r, D_W2r2, strides_none, "SAME")) + d_h2s)
    d_h3s = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h2, D_W3s, strides_down, "SAME")))
    d_h3r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h3s, D_W3r1, strides_none, "SAME")))
    d_h3 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h3r, D_W3r2, strides_none, "SAME")) + d_h3s)
    d_h4s = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h3, D_W4s, strides_down, "SAME")))
    d_h4r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h4s, D_W4r1, strides_none, "SAME")))
    d_h4 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h4r, D_W4r2, strides_none, "SAME")) + d_h4s)
    d_h5s = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h4, D_W5s, strides_down, "SAME")))
    d_h5r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h5s, D_W5r1, strides_none, "SAME")))
    d_h5 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h5r, D_W5r2, strides_none, "SAME")) + d_h5s)
    d_h6s = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h5, D_W6s, strides_down, "SAME")))
    d_h6r = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h6s, D_W6r1, strides_none, "SAME")))
    d_h6 = tf_tools.nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(d_h6r, D_W6r2, strides_none, "SAME")) + d_h6s)
    d_hf = tf.reshape(d_h6, [miniBatch_size, -1])
    if md_num_features > 0:  # project to get feature
        d_hf = tf_tools.nn_leaky_relu(tf.matmul(d_hf, D_Wmf) + D_bmf)
    if md_num_features != 0:  # use mbd
        d_hf = tf_tools.gan_minibatch(d_hf, D_WmT, D_bmT, md_num_kernels, md_kernel_dim)
    d_logit = tf.matmul(d_hf, D_Wo) + D_bo

    return d_logit
# ---------- End of Discriminator ----------


# ---------- Computational Graph ----------
# data feeding and augmentation
Z_ph = tf.placeholder(tf.float32, [miniBatch_size, noise_size])
X_ph = tf.placeholder(tf.float32, [miniBatch_size]+data_size+[1])
Y_ph = tf.placeholder(tf.float32, [miniBatch_size]+cond_size[1:3]+[3])
keep_prob_ph = tf.placeholder(tf.float32)

G_sample = generator(Z_ph, Y_ph)
D_logit_real = discriminator(X_ph, Y_ph)
D_logit_fake = discriminator(G_sample, Y_ph)

# supervised regularisation
if lambda_supervised > 0:
    if order_supervised == 2:
        tf.add_to_collection('g_losses', tf.div(tf.nn.l2_loss(G_sample-X_ph)*lambda_supervised, miniBatch_size))
    elif order_supervised == 1:
        tf.add_to_collection('g_losses', tf.reduce_mean(tf.reshape(tf.abs(G_sample-X_ph)*lambda_supervised, [miniBatch_size, -1])))
    elif order_supervised == -1:  # experimental KL divergence
        print('Nothing implemented yet!')
elif lambda_supervised < 0:  # experimental L2 loss only
    tf.add_to_collection('g_losses', tf.div(tf.nn.l2_loss(G_sample - X_ph), miniBatch_size))

# normal GAN-losses
tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real, tf.scalar_mul(.9, tf.ones_like(D_logit_real)))))
tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake))))
if lambda_supervised >= 0:
    tf.add_to_collection('g_losses', tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake))))

# optional conditioner loss here

G_loss = tf.add_n(tf.get_collection('g_losses'))
G_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(G_loss, var_list=theta_G)
D_loss = tf.add_n(tf.get_collection('d_losses'))
D_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(D_loss, var_list=theta_D)

sess = tf.Session()
sess.run(tf.initialize_all_variables())  # tf.global_variables_initializer

# train
for step in range(totalIterations):
    current_time = time.asctime(time.gmtime())

    if step in range(0, totalIterations, num_miniBatch):
        random.shuffle(trainDataIndices)
        if step < num_miniBatch * log_num_shuffle:
            print('trainDataIndices: %s' % trainDataIndices, flush=True, file=fid_output_info)

    miniBatch_idx = step % num_miniBatch
    miniBatch_indices = trainDataIndices[miniBatch_idx*miniBatch_size:(miniBatch_idx+1)*miniBatch_size]
    if step < log_num_shuffle * log_num_shuffle:
        print('miniBatch_indices: %s' % miniBatch_indices, flush=True, file=fid_output_info)

    X_train, Y_train = data_feeder.get_batch(miniBatch_indices)
    Z_train = noise_feeder.get_batch()

    D_nextTrainFeed = {X_ph: X_train, Y_ph: Y_train, Z_ph: Z_train, keep_prob_ph: keep_prob_rate}
    _, D_loss_value = sess.run([D_train_op, D_loss], feed_dict=D_nextTrainFeed)
    G_nextTrainFeed = {X_ph: X_train, Z_ph: Z_train, Y_ph: Y_train, keep_prob_ph: keep_prob_rate}
    _, G_loss_value = sess.run([G_train_op, G_loss], feed_dict=G_nextTrainFeed)

    C_loss_value = -1  # compatibility

    print('[%s] Step %d: d-loss=%f, g-loss=%f, c-loss=%f' % (current_time, step, D_loss_value, G_loss_value, C_loss_value))
    print('[%s] Step %d: d-loss=%f, g-loss=%f, c-loss=%f' % (current_time, step, D_loss_value, G_loss_value, C_loss_value), flush=True, file=fid_output_info)

    # Debug data
    if step in range(log_start_debug, totalIterations, log_freq_debug):
        # save debug samples
        filename_log_data = "debug_data_i%09d.h5" % step
        fid_debug_data = h5py.File(os.path.join(dir_output, filename_log_data), 'w')
        fid_debug_data.create_dataset('/indices_train/', data=miniBatch_indices)
        # fid_debug_data.create_dataset('/data_train/', X_train.shape, dtype=X_train.dtype, data=X_train)
        # fid_debug_data.create_dataset('/cond_train/', Y_train.shape, dtype=Y_train.dtype, data=Y_train)
        for repeat_idx in range(num_repeat_debug):
            # 1 - randomly sample conditions from training data
            Z_train = noise_feeder.get_batch()
            G_EvaluateFeed = {Z_ph: Z_train, Y_ph: Y_train, keep_prob_ph: 1}
            # test_t0 = time.time()
            samples = sess.run(G_sample, feed_dict=G_EvaluateFeed)
            # print('Elapsed time: %f second(s).' % (time.time() - test_t0), flush=True, file=fid_output_info)
            fid_debug_data.create_dataset('/samples_train_rep%03d/' % repeat_idx, samples.shape, dtype=samples.dtype, data=samples)

        fid_debug_data.create_dataset('/indices_dev/', data=devDataIndices)
        for repeat_idx in range(num_repeat_dev):
            # 2 - randomly sample conditions from dev (test) data
            miniBatch_indices_dev = [devDataIndices[i] for i in range(miniBatch_size*repeat_idx, miniBatch_size*(repeat_idx+1))]
            Y_dev = data_feeder.get_batch_grid_only(miniBatch_indices_dev)
            Z_dev = noise_feeder.get_batch()
            G_EvaluateFeed = {Z_ph: Z_dev, Y_ph: Y_dev, keep_prob_ph: 1}
            samples = sess.run(G_sample, feed_dict=G_EvaluateFeed)
            fid_debug_data.create_dataset('/samples_dev_rep%03d/' % repeat_idx, samples.shape, dtype=samples.dtype, data=samples)

        random.shuffle(noiseDimIndices)
        fid_debug_data.create_dataset('/dims_noise/', data=noiseDimIndices)
        samples_devDataIndices = (np.random.choice(devDataIndices, size=num_repeat_linspace, replace=False)).tolist()  # samples from dev data
        fid_debug_data.create_dataset('/data_noise/', data=samples_devDataIndices)
        for repeat_idx in range(num_repeat_linspace):
            Z_noise = noise_feeder.get_batch_linspace_samples(noiseDimIndices[repeat_idx])
            Y_noise = np.concatenate([data_feeder.get_batch_grid_only([samples_devDataIndices[repeat_idx]]) for i in range(miniBatch_size)], axis=0)
            G_EvaluateFeed = {Z_ph: Z_noise, Y_ph: Y_noise, keep_prob_ph: 1}
            samples = sess.run(G_sample, feed_dict=G_EvaluateFeed)
            fid_debug_data.create_dataset('/samples_noise_dim%03d/' % repeat_idx, samples.shape, dtype=samples.dtype, data=samples)

        # flush in the end
        fid_debug_data.flush()
        fid_debug_data.close()
        # NB. do not use continue here!
# ---------- End of Computational Graph ----------

fid_output_info.close()
