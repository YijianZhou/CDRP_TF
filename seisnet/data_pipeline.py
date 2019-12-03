"""classes and functions to write, read and feed data.
"""
import os, glob
import numpy as np
import tensorflow as tf

class Writer(object):
  """ TFRecords Writer
  Inputs:
    out_path - output path for TFRecords files
    data - training data in np.array
    det_label & ppk_label
  """

  def __init__(self, out_path):
    self._writer = tf.python_io.TFRecordWriter(out_path)

  def write(self, data, det_label, ppk_label):
    """
    write TFRecords
    """
    # encode
    example = tf.train.Example(features = tf.train.Features(feature={
                'data':      self._bytes_feature(data.tobytes()),
                'det_label': self._int64_feature(det_label),
                'ppk_label': self._bytes_feature(ppk_label.tobytes()) }))
    self._writer.write(example.SerializeToString())

  def _int64_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _bytes_feature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def close(self):
    self._writer.close()


class Reader(object):
  """ TFRecords Reader
  """
  def __init__(self, data_dir, data_shape):
    self._data_dir    = data_dir
    self._data_shape  = data_shape
    self._reader    = tf.TFRecordReader()

  def read(self):
    # setup shape
    num_steps = self._data_shape[0]
    step_len  = self._data_shape[1]
    num_chns  = self._data_shape[2]
    
    data_paths = os.path.join(self._data_dir, '*.tfrecords')
    fnames = glob.glob(data_paths)
    fname_q = tf.train.string_input_producer(fnames,
                                             shuffle = True,
                                             num_epochs = None)
    _, serial_eg = tf.TFRecordReader().read(fname_q)
    features = tf.parse_single_example(
                 serial_eg,
                 features={
                   'data':      tf.FixedLenFeature([], tf.string),
                   'det_label': tf.FixedLenFeature([], tf.int64),
                   'ppk_label': tf.FixedLenFeature([], tf.string),
                   })

    # decode data
    data = tf.decode_raw(features['data'], tf.float32)
    data.set_shape([num_steps * step_len * num_chns])
    data = tf.reshape(data, [num_steps, step_len, num_chns])

    # decode ppk label
    ppk_label = tf.decode_raw(features['ppk_label'], tf.float64)
    ppk_label.set_shape([num_steps])
    ppk_label = tf.reshape(ppk_label, [num_steps])

    # Pack
    features['data'] = data
    features['ppk_label'] = ppk_label
    return features


class Feeder(object):
  """ Feed training samples
  """

  def __init__(self, data_dir, data_shape):
    self._data_dir   = data_dir
    self._data_shape = data_shape # [batch_size, num_steps, win_len, num_chns]
    self._bsize      = self._data_shape[0]

    min_after_dequeue = 20000#TODO
    capacity = 20000 + 10 * 64

    self._reader = Reader(self._data_dir, self._data_shape[1:])
    samples   = self._reader.read()
    data      = samples["data"]
    det_label = samples["det_label"]
    ppk_label = samples["ppk_label"]

    self.data, self.det_label, self.ppk_label = tf.train.shuffle_batch(
                                                  [data, det_label, ppk_label],
                                                  batch_size = self._bsize,
                                                  capacity = capacity,
                                                  min_after_dequeue = min_after_dequeue)

