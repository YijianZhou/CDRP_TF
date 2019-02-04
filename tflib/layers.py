"""
Defination of layers
"""
import tensorflow as tf
import numpy as np

# 1D conv layer
def conv1d(inputs,
           num_filters,
           filter_width,
           stride = 1,
           padding = 'SAME',
           activation_fn = tf.nn.relu,
           initializer = tf.contrib.layers.variance_scaling_initializer(),
           scope = None):

  with tf.variable_scope(scope):
    # create conv filter weights
    num_chns = inputs.get_shape().as_list()[-1]
    filters = tf.get_variable(
                'filters',
                shape = [filter_width, num_chns, num_filters],
                initializer = initializer,
                collections = [tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    current_layer = tf.nn.conv1d(inputs, filters, stride, padding)
    # add bias
    biases = tf.get_variable(
               'biases',
                shape = [num_filters,],
                initializer = tf.constant_initializer(0.0),
                collections = [tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    current_layer = tf.nn.bias_add(current_layer, biases)
    # activation
    current_layer = activation_fn(current_layer)    
    return current_layer


# Fully-connected layer
def fc(inputs, 
       num_filters,
       activation_fn = tf.nn.relu,
       initializer = tf.contrib.layers.variance_scaling_initializer(),
       scope=None):

  with tf.variable_scope(scope):
    n_in = inputs.get_shape().as_list()[-1] # number of chns in
    weights = tf.get_variable(
                'weights',
                shape=[n_in, num_filters],
                initializer=initializer,
                collections = [tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    current_layer = tf.matmul(inputs, weights)

    biases = tf.get_variable(
               'biases',
               shape=[num_filters,],
               initializer=tf.constant_initializer(0.0),
               collections = [tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    current_layer = tf.nn.bias_add(current_layer, biases)
  
  return current_layer


# Bi-directional GRU layer
def bi_gru(inputs,
           num_layers = 2,
           num_units  = 32,
           initializer = tf.contrib.layers.variance_scaling_initializer(),
           scope=None):

  with tf.variable_scope(scope, initializer=initializer):
    fw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)])
    bw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
                                                      inputs, dtype=tf.float32)
  return outputs, states


# Batch-normalization
def batch_norm(inputs, scope=None, is_training=False):
  with tf.variable_scope(scope):
    return tf.layers.batch_normalization(inputs,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        training=is_training,
        trainable=True,
        name=scope,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,)


# Max-pooling layer
def max_pooling(inputs, ksize=2, stride=2):
    inputs = tf.expand_dims(inputs,2)
    outputs = tf.nn.max_pool(inputs, 
                            ksize=[1, ksize, 1, 1], 
                            strides=[1, stride, 1, 1], 
                            padding='SAME')
    outputs = tf.squeeze(outputs,squeeze_dims=[2])
    return outputs
