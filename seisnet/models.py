""" Defination of models
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/CDRP_TF')

import numpy as np
import tensorflow as tf
# import model
import config
import tflib.layers as layers
from tflib.nn_model import BaseModel


# CNN for Earthquake Detection
class DetNet(object):

  def __init__(self, inputs, ckpt_dir):
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.inputs = inputs
    self.data = tf.cond(self.is_training, 
                  lambda:self.inputs[0]['data'], lambda:self.inputs[1]['data'])
    self.ckpt_dir = ckpt_dir
    self.layers   = {}


  def _setup_prediction(self):

    # config model
    cfg = config.Config()
    self.config = cfg
    # model structure
    num_layers   = cfg.num_cnn_layers
    num_filters  = cfg.num_cnn_filters
    filter_width = cfg.filter_width
    # train params
    self.config.lrate = cfg.cnn_lrate
    lamb  = cfg.cnn_l2reg
    self.ckpt_step = cfg.ckpt_step
    self.summary_step = cfg.summary_step
    # inputs
    current_layer = self.data
    current_layer = tf.squeeze(current_layer, [1])
    bsize, _, _ = current_layer.get_shape().as_list()

    # model prediction
    for i in range(num_layers):
        current_layer = layers.conv1d(current_layer, num_filters, filter_width,
                                      scope='conv{}'.format(i+1))
        current_layer = layers.batch_norm(current_layer, 
                                          scope='batch_norm{}'.format(i+1))
        current_layer = tf.nn.relu(current_layer)
        current_layer = layers.max_pooling(current_layer)
        self.layers['conv{}'.format(i+1)] = current_layer

    # fully connected layer
    current_layer = tf.reshape(current_layer, [bsize,-1], name='reshape')
    current_layer = layers.fc(current_layer, 2, scope='logits')
    self.layers['logits'] = current_layer
    # softmax regression
    current_layer = tf.nn.softmax(current_layer)
    self.layers['pred_prob'] = current_layer
    current_layer = tf.argmax(current_layer, 1)
    self.layers['pred_class'] = current_layer

    # L2 regularization
    tf.contrib.layers.apply_regularization(
      tf.contrib.layers.l2_regularizer(lamb),
      weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))


  # setup loss function
  def _setup_loss(self):

    # setup label
    self.label = tf.cond(self.is_training,
                   lambda:self.inputs[0]['label'], lambda:self.inputs[1]['label'])
    bsize = self.label.get_shape().as_list()[0]
    pos_labels = self.label[0:bsize/2]
    neg_labels = self.label[bsize/2:]
    pred_class = self.layers['pred_class']

    # calc loss
    with tf.name_scope('loss'):
        labels = self.label
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.layers['logits'], 
                            labels=labels)
        raw_loss = tf.reduce_mean(cross_entropy)
        self.loss  = raw_loss
        with tf.name_scope('regularization'):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss   = tf.reduce_sum(reg_losses)
        self.loss += reg_loss

    # calc accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(labels, pred_class)
        self.accuracy = tf.reduce_mean(tf.to_float(correct_pred))
        correct_pos = tf.equal(pos_labels, pred_class[0:bsize/2])
        self.pos_acc = tf.reduce_mean(tf.to_float(correct_pos))
        correct_neg = tf.equal(neg_labels, pred_class[bsize/2:])
        self.neg_acc = tf.reduce_mean(tf.to_float(correct_neg))

    # add summary
    tf.summary.scalar('raw_loss', raw_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    tf.summary.scalar('events_acc', self.pos_acc)
    tf.summary.scalar('noise_acc',  self.neg_acc)


  # def optimizer
  def _setup_optimizer(self, learning_rate, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    updates = tf.group(*update_ops, name='update_ops')
    with tf.control_dependencies([updates]):
        self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(
              self.loss, global_step=global_step)
    self.optimizer = optim



# RNN for Phase Picking
class PpkNet(object):

  def __init__(self, inputs, ckpt_dir):
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.inputs = inputs
    self.data = tf.cond(self.is_training,
                  lambda:self.inputs[0]['data'],  lambda:self.inputs[1]['data'])
    self.ckpt_dir = ckpt_dir
    self.layers   = {}


  def _setup_prediction(self):

    # config model
    cfg = config.Config()
    self.config = cfg
    self.ckpt_step = cfg.ckpt_step
    self.summary_step = cfg.summary_step
    # model structure
    num_units  = cfg.num_units
    num_layers = cfg.num_rnn_layers
    # training params
    self.config.lrate = cfg.rnn_lrate
    self.ckpt_step = cfg.ckpt_step
    self.summary_step = cfg.summary_step
    # input data
    current_layer = self.data

    # model prediction
    bsize, num_step, _, _ = current_layer.get_shape().as_list()
    current_layer = tf.reshape(current_layer, [bsize, num_step, -1])
    output0, state = layers.bi_gru(current_layer,
                                   num_layers = num_layers,
                                   num_units  = num_units,
                                   scope='bi_gru')
    output = tf.concat([output0[0], output0[1]], axis=2)
    output = tf.reshape(output, [-1, 2*num_units]) # flatten bi-rnn
    logits = layers.fc(output, 3, scope='ppk_logits', activation_fn=None) # [0 1 2] for [N, P, S] 
    self.layers['logits'] = logits

    flat_prob = tf.nn.softmax(logits, name='pred_prob')
    self.layers['pred_prob']  = tf.reshape(flat_prob, [-1, num_step, 3]) # [bsize, num_step, num_chns]
    self.layers['pred_class'] = tf.argmax(self.layers['pred_prob'], 2, name='pred_class')


  def _setup_loss(self):

    self.label = tf.cond(self.is_training,
                   lambda:self.inputs[0]['label'], lambda:self.inputs[1]['label'])
    # calc loss
    with tf.name_scope('loss'):
        labels = tf.to_int64(self.label) # [batch_size, num_steps]
        flat_labels = tf.reshape(labels, [-1])
        # cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                            logits = self.layers['logits'],
                            labels = flat_labels)
        self.loss = tf.reduce_mean(cross_entropy)

    # calc accuracy
    with tf.name_scope('accuracy'):
        correct_ppk = tf.to_float(tf.equal(\
                          labels, self.layers['pred_class']))
        self.accuracy = tf.reduce_mean(correct_ppk)

    # add summary
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)


  def _setup_optimizer(self, learning_rate, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if  update_ops:
        updates = tf.group(*update_ops, name='update_ops')
        with tf.control_dependencies([updates]):
            self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate)
    self.optimizer = optim.minimize(self.loss,
                                    name = 'optimizer',
                                    global_step = global_step)
