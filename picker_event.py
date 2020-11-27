""" CDRP picker for event data
Inputs
  ckpt_dir: root dir for checkpoint
  ckpt_step: step of checkpoint for PpkNet
"""
import os, sys, glob, time
sys.path.append('/home/zhouyj/software/CDRP_TF')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from obspy import read, UTCDateTime
# import model
import seisnet.models as models
import seisnet.config as config
import seisnet.data_pipeline as dp
from tflib.nn_model import BaseModel
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)


class CDRP_Picker_Event(object):

  def __init__(self, 
               ckpt_dir = '/home/zhouyj/software/CDRP_TF/output/tmp/PpkNet',
               ckpt_step = None):

    self.ckpt_dir = ckpt_dir
    self.ckpt_step = ckpt_step
    cfg = config.Config()
    self.samp_rate = cfg.sampling_rate
    self.win_len = int(self.samp_rate * cfg.win_len)
    self.step_len = int(self.samp_rate * cfg.step_len)
    self.step_stride = int(self.samp_rate * cfg.step_stride)
    self.num_steps = int(-(cfg.step_len / cfg.step_stride - 1) + \
                         cfg.win_len / cfg.step_stride)


  def pick(self, streams):
    """ run PpkNet
    """
    picks = []
    data_batch = self.fetch_data(streams, self.num_steps, self.step_len, self.step_stride)
    data_holder = tf.placeholder(tf.float32, shape=data_batch.shape)
    inputs = [{'data':data_holder},{'data':data_holder}]
    with tf.Session() as sess:
        # set up PpkNet model
        model = models.PpkNet(inputs, self.ckpt_dir)
        BaseModel(model).load(sess, self.ckpt_step)
        to_fetch = model.layers['pred_class']
        # run PpkNet
        feed_dict = {inputs[1]['data']: data_batch,
                     model.is_training: False}
        run_time_start = time.time()
        pred_classes = sess.run(to_fetch, feed_dict)
        ppk_time = time.time() - run_time_start
        # decode to sec
        for pred_class in pred_classes:
            pred_p = np.where(pred_class==1)[0]
            if len(pred_p)>0:
                tp = self.step_len/2 if pred_p[0]==0 \
                else self.step_len + self.step_stride * (pred_p[0]-0.5)
                tp /= self.samp_rate
                pred_class[0:pred_p[0]] = 0
            else: tp = -1
            pred_s = np.where(pred_class==2)[0]
            if len(pred_s)>0:
                ts = self.step_len/2 if pred_s[0]==0 \
                else self.step_len + self.step_stride * (pred_s[0]-0.5)
                ts /= self.samp_rate
            else: ts = -1
            picks.append([tp, ts])
    tf.reset_default_graph()
    return picks


  def fetch_data(self, streams, num_steps, step_len, step_stride):
    batch_size = len(streams)
    data_holder = np.zeros((batch_size, num_steps, step_len, 3), dtype=np.float32)
    for i,stream in enumerate(streams):
        # convert to numpy array
        st_data = np.array([trace.data for trace in stream], dtype=np.float32)
        # feed into holder
        for j in range(num_steps):
            idx0 = j * step_stride
            idx1 = idx0 + step_len
            if idx1>st_data.shape[1]: continue
            current_step = st_data[:, idx0:idx1]
            data_holder[i, j, :, :] = np.transpose(current_step)
    return data_holder

