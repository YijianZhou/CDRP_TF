""" CDRP picker for continuous data (stream)
Inputs
  out_file: out_file = open('file_name', 'w')
  ckpt_dir: root dir for checkpoint
  cnn_ckpt_step: step of checkpoint for DetNet
  rnn_ckpt_step: step of checkpoint for PpkNet
"""
import os, sys, glob, time
sys.path.append('/home/zhouyj/software/CDRP_TF/')
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

class CDRP_Picker_Stream(object):

  def __init__(self,
               cnn_ckpt_dir = '/home/zhouyj/software/CDRP_TF/output/tmp/DetNet',
               rnn_ckpt_dir = '/home/zhouyj/software/CDRP_TF/output/tmp/PpkNet',
               cnn_ckpt_step = None,
               rnn_ckpt_step = None):
    self.cnn_ckpt_dir = cnn_ckpt_dir
    self.rnn_ckpt_dir = rnn_ckpt_dir
    self.cnn_ckpt_step = cnn_ckpt_step
    self.rnn_ckpt_step = rnn_ckpt_step
    self.config = config.Config()
    self.samp_rate = self.config.sampling_rate
    self.freq_band = self.config.freq_band
    self.win_len = self.config.win_len # sec
    self.win_len_npts = int(self.samp_rate*self.win_len)
    self.step_len = self.config.step_len
    self.step_len_npts = int(self.step_len * self.samp_rate)
    self.step_stride = self.config.step_stride
    self.step_stride_npts = int(self.step_stride * self.samp_rate)
    self.num_steps = int(-(self.step_len/self.step_stride-1) +\
                           self.win_len/self.step_stride)


  def pick(self, stream, out_file):
    """ pick stream
    Inputs:
      stream: obspy.core.Stream, with 3 chn (shape=[3,win_len])
    """
    # run DetNet & PpkNet
    det_list = self.run_det(stream)
    self.run_ppk(stream, det_list, out_file)


  def run_det(self, stream):
    """ run DetNet to detect events in continuous stream
    """
    run_time_start = time.time()
    # check stream data
    if len(stream)!=3: print('missing trace!'); return []
    # get header info
    start_time = max([trace.stats.starttime for trace in stream])
    end_time = min([trace.stats.endtime for trace in stream])
    if end_time < start_time + self.win_len: return  []
    # sliding window with half overlap
    num_steps = int((end_time-start_time) / (self.win_len/2)) -1
    det_list=[]
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # import DetNet
        inputs = {'data': tf.placeholder(tf.float32,
                            shape = (1, 1, self.win_len_npts, 3))}
        inputs = [inputs, inputs]
        model = models.DetNet(inputs, self.cnn_ckpt_dir)
        BaseModel(model).load(sess, self.cnn_ckpt_step)
        to_fetch = [model.layers['pred_class'],
                    model.layers['pred_prob']]
        num_events = 0
        for step_idx in range(num_steps):
            # get time range
            t0 = start_time + step_idx*self.win_len/2
            t1 = t0 + self.win_len
            # run DetNet
            st = self.preprocess(stream.slice(t0, t1).copy())
            feed_dict = {inputs[1]['data']: self.fetch_data(st, 
                         1, self.win_len_npts, self.win_len_npts),
                         model.is_training: False}
            [pred_class, pred_prob] = sess.run(to_fetch, feed_dict)
            is_event = pred_class[0] > 0
            if is_event:
                num_events += 1
                det_list.append([t0, t1, pred_prob[0][1]])
                print('detected events: {} to {} ({:.2f}%)'.\
                format(t0, t1, pred_prob[0][1]*100))
    print("processed {} windows".format(num_steps))
    print("DetNet Run time: {:.2f}s".format(time.time() - run_time_start))
    print("found {} events".format(num_events))
    tf.reset_default_graph()
    return det_list


  def run_ppk(self, stream, det_list, out_file):
    """ run PpkNet to ppk the detected events
    """
    run_time_start = time.time()
    if len(det_list)==0: return
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # set up PpkNet model
        inputs = {'data': tf.placeholder(tf.float32,
                            shape = (1, self.num_steps, self.step_len_npts, 3))}
        inputs = [inputs, inputs]
        model = models.PpkNet(inputs, self.rnn_ckpt_dir)
        BaseModel(model).load(sess, self.rnn_ckpt_step)
        to_fetch = model.layers['pred_class']
        # for all dets
        num_events=0
        old_t1 = det_list[0][0]
        for idx, det in enumerate(det_list):
            t0, t1, det_prob = det[0], det[1], det[2]
            # pick the time windows with P in first half
            # if (1) no consecutive picks
            # or (2.1) next win is event
            #    (2.2) and with higher pred_prob
            new_idx = min(idx+1, len(det_list)-1)
            if t0 < old_t1 \
              or (t1 > det_list[new_idx][0] \
                and det_list[idx][2] < det_list[new_idx][2]): 
                continue
            else:
                # run PpkNet
                st = self.preprocess(stream.slice(t0,t1).copy())
                feed_dict = {inputs[1]['data']: self.fetch_data(st,
                             self.num_steps, self.step_len_npts, self.step_stride_npts),
                             model.is_training: False}
                pred_class = sess.run(to_fetch, feed_dict)[0]
                # decode to relative time (sec) to win_t0
                pred_p = np.where(pred_class==1)[0]
                if len(pred_p)>0:
                    if pred_p[0]==0: tp = t0 + self.step_len/2
                    else: tp = t0 + self.step_len + self.step_stride*(pred_p[0]-0.5)
                    pred_class[0:pred_p[0]] = 0
                else: tp = -1
                pred_s = np.where(pred_class==2)[0]
                if len(pred_s)>0:
                    if pred_s[0]==0: ts = t0 + self.step_len/2
                    else: ts = t0 + self.step_len + self.step_stride*(pred_s[0]-0.5)
                else: ts = -1
                print('picked phase time: tp={}, ts={}'.format(tp, ts))
                out_file.write('{},{},{}\n'.format(stream[0].stats.station, tp, ts))
                num_events += 1
                old_t1 = t1 # if picked
    print("Picked {} events".format(num_events))
    print("PpkNet Run time: {:.2f}s".format(time.time() - run_time_start))
    tf.reset_default_graph()
    return


  def fetch_data(self, stream, num_steps, step_len, step_stride):
    # get stream data
    data_holder = np.zeros((1, num_steps, step_len, 3), dtype=np.float32)
    st_data = np.array([trace.data for trace in stream], dtype=np.float32)
    # feed into time steps
    for i in range(num_steps):
        idx_0 = i * step_stride
        idx_1 = idx_0 + step_len
        if idx_1>st_data.shape[1]: continue
        current_step = st_data[:, idx_0:idx_1]
        data_holder[0, i, :, :] = np.transpose(current_step)
    return data_holder


  def preprocess(self, st):
    # resample data
    org_rate = int(st[0].stats.sampling_rate)
    rate = np.gcd(org_rate, int(self.samp_rate))
    if rate==1: print('warning: bad sampling rate!'); return []
    decim_factor = int(org_rate / rate)
    resamp_factor = int(self.samp_rate / rate)
    if decim_factor!=1: st = st.decimate(decim_factor)
    if resamp_factor!=1: st = st.interpolate(self.samp_rate)
    # filter
    st = st.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
    flt_type, freq_rng = self.freq_band
    if flt_type=='highpass':
        return st.filter(flt_type, freq=freq_rng).normalize()
    if flt_type=='bandpass':
        return st.filter(flt_type, freqmin=freq_rng[0], freqmax=freq_rng[1]).normalize()

