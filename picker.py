""" Apply trained model as picker method
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

class CDRP_Picker(object):

  def __init__(self, out_file,
               ckpt_dir = '/home/zhouyj/software/CDRP_TF/output/tmp',
               cnn_ckpt_step = None,
               rnn_ckpt_step = None):

    self.out_file = out_file
    self.cnn_ckpt_dir = os.path.join(ckpt_dir, 'DetNet')
    self.rnn_ckpt_dir = os.path.join(ckpt_dir, 'PpkNet')
    self.cnn_ckpt_step = cnn_ckpt_step
    self.rnn_ckpt_step = rnn_ckpt_step
    self.config = config.Config()
    self.win_len = self.config.win_len
    self.step_len = self.config.step_len
    self.step_stride = self.config.step_stride
    self.num_steps = int(-(self.step_len/self.step_stride-1) +\
                           self.win_len/self.step_stride)


  def pick(self, stream):
    """ pick stream
    Inputs:
      stream: obspy.core.Stream, with 3 chn (shape=[3,win_len])
    """
    # read stream
    det_list = self.run_det(stream)
    self.run_ppk(stream, det_list)


  def run_det(self, stream):
    """ run DetNet to detect events in continuous stream
    """

    # check stream data
    if len(stream)!=3: print('missing trace!'); return []
    # get header info
    hd0 = stream[0].stats
    hd1 = stream[1].stats
    hd2 = stream[2].stats
    start_time = max(hd0.starttime, hd1.starttime, hd2.starttime)
    end_time   = min(hd0.endtime,   hd1.endtime,   hd2.endtime)
    if end_time < start_time + self.win_len/100: return  []
    # make time sequence
    num_win = int((end_time - start_time) /self.win_len/2) -1
    time_seq = np.arange(0,num_win*self.win_len, self.win_len)

    det_list=[]
    with tf.Session() as sess:
        # import DetNet
        win_point_len = int(100*self.win_len)
        inputs = {'data': tf.placeholder(tf.float32,
                            shape = (1, 1, win_point_len, 3))}
        inputs = [inputs, inputs]
        model = models.DetNet(inputs, self.cnn_ckpt_dir)
        BaseModel(model).load(sess, self.cnn_ckpt_step)
        to_fetch = [model.layers['pred_class'],
                     model.layers['pred_prob']]

        num_events = 0
        run_time_start = time.time()
        for dt in time_seq:

            # get time range
            t0 = start_time + dt
            t1 = t0 + self.win_len
            # run DetNet
            st = self.preprocess(stream.slice(t0, t1))
            feed_dict = {inputs[1]['data']: self.fetch_data(st,
                                           1, win_point_len, win_point_len),
                         model.is_training: False}
            [pred_class, pred_prob] = sess.run(to_fetch, feed_dict)
            
            is_event = pred_class[0] > 0
            if is_event:
                num_events += 1
                det_list.append([t0, t1, pred_prob[0][1]])
                print('detected events: {} to {} ({:.2f}%)'.\
                format(t0, t1, pred_prob[0][1]*100))

    print("processed {} windows".format(len(time_seq)))
    print("DetNet Run time: ", time.time() - run_time_start)
    print("found {} events".format(num_events))
    tf.reset_default_graph()
    return det_list


  def run_ppk(self, stream, det_list):
    """ run PpkNet to ppk the detected events
    """

    with tf.Session() as sess:
        # set up PpkNet model
        step_point_len = int(100*self.step_len)
        step_point_stride = int(100*self.step_stride)
        inputs = {'data': tf.placeholder(tf.float32,
                            shape = (1, self.num_steps, step_point_len, 3))}
        inputs = [inputs, inputs]
        model = models.PpkNet(inputs, self.rnn_ckpt_dir)
        BaseModel(model).load(sess, self.cnn_ckpt_step)
        to_fetch = model.layers['pred_class']

        run_time_start = time.time()
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
                st = self.preprocess(stream.slice(t0,t1))
                feed_dict = {inputs[1]['data']: self.fetch_data(st,
                               self.num_steps, step_point_len, step_point_stride),
                             model.is_training: False}
                pred_class = sess.run(to_fetch, feed_dict)[0]
                
                # decode to relative time (sec) to win_t0
                pred_p = np.where(pred_class==1)[0]
                pred_s = np.where(pred_class==2)[0]
                if len(pred_p)>0:
                    if pred_p[0]==0: tp = t0 + self.step_len/2
                    else:            tp = t0 + self.step_len + self.step_stride *(pred_p[0]-0.5)
                else: tp = -1
                if len(pred_s)>0:
                    if pred_s[0]==0: ts = t0 + self.step_len/2
                    else:            ts = t0 + self.step_len + self.step_stride *(pred_s[0]-0.5)
                else: ts = -1
                
                print('picked phase time: tp={}, ts={}'.format(tp, ts))
                self.out_file.write(unicode('{},{},{}\n'.\
                                    format(stream[0].stats.station, tp, ts)))
                num_events += 1
                old_t1 = t1 # if picked
    
    print("Picked {} events".format(num_events))
    print("PpkNet Run time: ", time.time() - run_time_start)
    tf.reset_default_graph()
    return


  def fetch_data(self, stream, num_steps, step_len, step_stride):

    # convert to numpy array
    xdata = np.float32(stream[0].data)
    ydata = np.float32(stream[1].data)
    zdata = np.float32(stream[2].data)
    st_data = np.array([xdata, ydata, zdata])

    # feed into time steps
    time_steps = np.zeros((1, num_steps, step_len, 3), dtype=np.float32)
    for i in range(num_steps):
        idx_0 = i * step_stride
        idx_1 = idx_0 + step_len
        current_step = st_data[:, idx_0:idx_1]
        time_steps[0, i, :, :] = np.transpose(current_step)
    return time_steps


  def preprocess(self, stream):
    """preprocess for CDRP model:
        rmean + rtr + normalize"""
    stream = stream.detrend('constant') # rmean + rtr in SAC
    stream = stream.filter('highpass', freq=1.)
    return stream.normalize()

