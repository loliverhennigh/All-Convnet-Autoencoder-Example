

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool_2x2(x, W):
  return tf.nn.conv2d(x, W,
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

# ins and outs
x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder("float") # do a little dropout to normalize
x_norm = tf.nn.dropout(x, keep_prob)
y_ = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x_norm, [-1, 28, 28, 1])
# Need the batch size for the transpose layers.
batch_size = tf.shape(x)[0]

# Define all the weight for the encoder part
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_pool1 = weight_variable([2, 2, 32, 32])
b_pool1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

W_pool2 = weight_variable([2, 2, 64, 64])
b_pool2 = bias_variable([64])

W_conv3 = weight_variable([1, 1, 64, 64])
b_conv3 = bias_variable([64])

W_conv4 = weight_variable([1, 1, 64, 5])
b_conv4 = bias_variable([5])

# Calc all layers 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.relu(pool_2x2(h_conv1, W_pool1) + b_pool1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf.nn.relu(pool_2x2(h_conv2, W_pool2) + b_pool2)
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv4 =tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

####################################################
## Now the we have the min layer as a 7x7x5 = 245 ##
####################################################

# Define all the weight for the decoder part
# and deconv_shape. Thats the tricky bit because
# you have to start thinking about the height 
# and width
W_conv5 = weight_variable([1, 1, 64, 5])
b_conv5 = bias_variable([64])
deconv_shape_conv5 = tf.pack([batch_size, 7, 7, 64])

W_pool3 = weight_variable([2, 2, 64, 64])
b_pool3 = bias_variable([64])
deconv_shape_pool3 = tf.pack([batch_size, 14, 14, 64])

W_conv6 = weight_variable([5, 5, 32, 64])
b_conv6 = bias_variable([32])
deconv_shape_conv6 = tf.pack([batch_size, 14, 14, 32])

W_pool4 = weight_variable([2, 2, 32, 32])
b_pool4 = bias_variable([32])
deconv_shape_pool4 = tf.pack([batch_size, 28, 28, 32])

W_conv7 = weight_variable([5, 5, 1, 32])
b_conv7 = bias_variable([1])
deconv_shape_conv7 = tf.pack([batch_size, 28, 28, 1])

# Now the conv2d_transpose part. Hopfuly just looking
# at the encoder part and decoder part side by side
# will make it clear how it works.
h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv4, W_conv5, output_shape = deconv_shape_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
h_pool3 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W_pool3, output_shape = deconv_shape_pool3, strides=[1,2,2,1], padding='SAME') + b_pool3)
h_conv6 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool3, W_conv6, output_shape = deconv_shape_conv6, strides=[1,1,1,1], padding='SAME') + b_conv6)
h_pool4 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W_pool4, output_shape = deconv_shape_pool4, strides=[1,2,2,1], padding='SAME') + b_pool4)
h_conv7 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool4, W_conv7, output_shape = deconv_shape_conv7, strides=[1,1,1,1], padding='SAME') + b_conv7)
 
y_conv = tf.reshape(h_conv7, [-1, 784])

error = tf.nn.l2_loss(y_ - y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(error) # I made the learning rate smaller then normal
accuracy = tf.nn.l2_loss(y_ - y_conv)
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%20 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_:batch[0], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("Saving test image to new_run_1.png")
    new_im = y_conv.eval(feed_dict={x: batch[0], y_: batch[0], keep_prob: 1.0})
    plt.imshow(new_im[1].reshape((28,28)))
    plt.savefig('new_run_1.png')
    print("Saved")
  train_step.run(feed_dict={x: batch[0], y_: batch[0], keep_prob: 0.8})


