
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional rgbd_10 model example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sklearn
import input_data
from matplotlib import pyplot as plt


# TODO
# These are some useful constants that you can use in your code.
# Feel free to ignore them or change them.
# TODO
IMAGE_SIZE = 32
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 1024
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
# This is where the data gets stored
#TRAIN_DIR = 'data'
# HINT:
# if you are working on the computers in the pool and do not want
# to download all the data you can use the pre-loaded data like this:
TRAIN_DIR = '/home/mllect/data/rgbd'


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])

def fake_data(num_images, channels):
  """Generate a fake dataset that matches the dimensions of rgbd_10 dataset."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, channels),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


#classic way, stride of 1 and padding makes it preserve the size of the image
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,seed = SEED)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    NUM_CHANNELS = 1
    train_data, train_labels = fake_data(256, NUM_CHANNELS)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE, NUM_CHANNELS)
    num_epochs = 1
  else:
    if (FLAGS.use_rgbd):
      NUM_CHANNELS = 4
      print('****** RGBD_10 dataset ******')
      print('* Input: RGB-D              *')
      print('* Channels: 4               *')
      print('*****************************')
    else:
      NUM_CHANNELS = 3
      print('****** RGBD_10 dataset ******')
      print('* Input: RGB                *')
      print('* Channels: 3               *')
      print('*****************************')
    # Load input data
    data_sets = input_data.read_data_sets(TRAIN_DIR, FLAGS.use_rgbd)
    num_epochs = NUM_EPOCHS

    train_data = data_sets.train.images  # (27999, 32, 32, 3)
    train_labels= data_sets.train.labels
  
    test_data = data_sets.test.images #(6321, 32, 32, 3)
    test_labels = data_sets.test.labels
  
    validation_data = data_sets.validation.images #(5413, 32, 32, 3)
    validation_labels = data_sets.validation.labels
  
  train_size = train_labels.shape[0]
 
  train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  keep_prob = tf.placeholder(tf.float32)


  #Get the predictions of data
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={eval_data: data[begin:end, ...],keep_prob:1})                    
      else:
        batch_predictions = sess.run(eval_prediction,feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...],keep_prob:1})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

    
  #Define the weights
  
  W_conv11 = weight_variable([3, 3, NUM_CHANNELS, 32])
  b_conv11 = bias_variable([32]) 
  W_conv12 = weight_variable([3, 3, 32, 32])
  b_conv12 = bias_variable([32])
  
  W_conv21 = weight_variable([3, 3, 32, 64])
  b_conv21 = bias_variable([64])
  W_conv22 = weight_variable([3, 3, 64, 64])
  b_conv22 = bias_variable([64])

  W_conv31 = weight_variable([3, 3, 64, 128])
  b_conv31 = bias_variable([128])
  W_conv32 = weight_variable([3, 3, 128, 128])
  b_conv32 = bias_variable([128])
  
  W_fc = weight_variable([4 * 4 * 128, 256])
  b_fc = bias_variable([256])
  
  W_out = weight_variable([256, NUM_LABELS])
  b_out = bias_variable([NUM_LABELS])

  #Define the architecture
  def conv_nn(x,keep_prob):
    ######First layer ########
    h_conv11 = conv2d(x, W_conv11) + b_conv11
    h_relu11 = tf.nn.relu(h_conv11)
    h_conv12 = conv2d(h_relu11, W_conv12) + b_conv12
    h_relu12 = tf.nn.relu(h_conv12)
    h_pool11 = max_pool_2x2(h_relu12)

    ##### Second layer #######
    h_conv21 = conv2d(h_pool11, W_conv21) + b_conv21
    h_relu21 = tf.nn.relu(h_conv21)
    h_conv22 = conv2d(h_relu21, W_conv22) + b_conv22
    h_relu22 = tf.nn.relu(h_conv22)    
    h_pool21 = max_pool_2x2(h_relu22)

    ##### Third layer #######
    h_conv31 = conv2d(h_pool21, W_conv31) + b_conv31
    h_relu31 = tf.nn.relu(h_conv31)
    h_conv32 = conv2d(h_relu31, W_conv32) + b_conv32
    h_relu32 = tf.nn.relu(h_conv32)    
    h_pool31 = max_pool_2x2(h_relu32)


    ###### Fully connected layer ####
    h_pool31_flat = tf.reshape(h_pool31, [-1, 4*4*128])
    h_fc = tf.nn.relu(tf.matmul(h_pool31_flat, W_fc) + b_fc)

    #dropout keep_prob = 0.5 in case of training and 1 otherwise
    h_fc = tf.nn.dropout(h_fc, keep_prob)

    ###### Output layer #########
    logits = tf.matmul(h_fc, W_out) + b_out
    return logits

  #Feed the inputs to the cnn
  logits = conv_nn(train_data_node,keep_prob)
  eval_logits = conv_nn(eval_data,keep_prob)

  #Get the predictions
  train_pred = tf.nn.softmax(logits)
  eval_prediction = tf.nn.softmax(eval_logits)
 
  # Compute the loss of the model
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,train_labels_node))

  #Regularization
  regularize = tf.nn.l2_loss(W_fc) + tf.nn.l2_loss(b_fc) + tf.nn.l2_loss(W_out) + tf.nn.l2_loss(b_out)
  loss+= 0.01*regularize

  #define an optimizer
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


  #Train
  with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      val_list = []
      train_list = []
      for epoch in range(NUM_EPOCHS):
          epoch_loss = 0
          epoch_train_error = 0
          for b in range(int(train_size/BATCH_SIZE)):  # There will be some data left out at the end !! fix it
              #getting the right batch for training
              batch_begin = b*BATCH_SIZE
              batch_end = batch_begin + BATCH_SIZE
              x_train_batch = train_data[batch_begin:batch_end]
              y_train_batch = train_labels[batch_begin:batch_end]

              #get the loss and optimize
              _,l =sess.run([optimizer,loss],feed_dict={train_data_node:x_train_batch, train_labels_node:y_train_batch,keep_prob:0.5})
              pred = sess.run(train_pred,feed_dict={train_data_node:x_train_batch, keep_prob:1})
              epoch_loss+=l
              train_error = error_rate(pred, y_train_batch)
              epoch_train_error+= train_error
              if((b%EVAL_FREQUENCY) == 0):
                  #Validatiion error should be on the whole validation data
                  val_pred = eval_in_batches(validation_data,sess)
                  val_error = error_rate(val_pred, validation_labels)
                  val_list.append(val_error)
                  train_list.append(train_error)
                  print('Epoch: %d, step: %d, loss: %.2f, train error: %.2f %%, validation error: %.2f %%' %
                        (epoch+1,b,l,train_error,val_error))   
          epoch_train_error = epoch_train_error/(b+1)
          print('Epoch %d completed out of %d, epoch_loss: %.2f, epoch_train_error: %.2f %%' %
                (epoch+1,NUM_EPOCHS,epoch_loss,epoch_train_error))
       
      test_pred = eval_in_batches(test_data,sess)
      test_error = error_rate(test_pred,test_labels)
      print('Test error: %.2f %%' % test_error)
      print('Confusion matrix:')
    #  NOTE: the following will require scikit-learn
      print(confusion_matrix(test_labels, numpy.argmax(eval_in_batches(test_data, sess), 1)))
      plt.axis([0, 60, 0, 100])
      plt.xlabel("steps")
      plt.ylabel("Error(%)")
      plt.plot(val_list,label = 'Validation error')
      plt.plot(train_list, label = 'Training error')
      plt.title("Training vs Validation error")
      plt.legend()
      plt.show()

 
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_rgbd',
      default=False,
      help='Use rgb-d input data (4 channels).',
      action='store_true'
  )
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()
