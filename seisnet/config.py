""" Configure file for CDRP_TF model
"""
class Config(object):

  def __init__(self):

    # defaults
    self.sampling_rate = 100
    self.num_chns = 3

    # cnn model
    self.win_len = 30. # in sec
    self.num_cnn_filters = 32
    self.filter_width = 3
    self.num_cnn_layers = 8
    self.cnn_l2reg = 1e-4

    # cnn train
    self.cnn_bsize = 128
    self.cnn_lrate = 1e-4
    self.ckpt_step = 25
    self.summary_step = 5

    # rnn model
    self.step_len = 1. # in sec
    self.step_stride = self.step_len/2
    self.num_units  = 64
    self.num_rnn_layers = 2

    # rnn traing
    self.rnn_bsize = 128
    self.rnn_lrate = 1e-3

