""" Pick Gorkha data with CDRP
"""
import os, sys, glob, shutil
sys.path.append('/home/zhouyj/software/CDRP_TF/')
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
import multiprocessing as mp
from obspy import read, UTCDateTime
import picker
from reader import get_xq_data

# i/o paths
data_root = '/data1/Gorkha_raw'
get_data_dict = get_xq_data
time_range = '20150616-20160516'
out_root = '/home/zhouyj/Gorkha/run_cdrp/output/picks_cdrp'
if not os.path.exists(out_root): os.makedirs(out_root)
# set picker
cdrp_dir = '/home/zhouyj/software/CDRP_TF'
shutil.copyfile('config_gorkha.py', os.path.join(cdrp_dir,'seisnet','config.py'))
cnn_ckpt_dir = '/home/zhouyj/Gorkha/run_cdrp/output/gorkha_det/DetNet'
rnn_ckpt_dir = '/home/zhouyj/Gorkha/run_cdrp/output/rc_ppk/PpkNet'
cnn_ckpt_step = 23000
rnn_ckpt_step = 9500
picker = picker.CDRP_Picker(cnn_ckpt_dir=cnn_ckpt_dir, rnn_ckpt_dir=rnn_ckpt_dir, 
    cnn_ckpt_step=cnn_ckpt_step, rnn_ckpt_step=rnn_ckpt_step)

def pick_day(date):
    t0_code = ''.join(str(date.date).split('-'))
    t1_code = ''.join(str((date+86400).date).split('-'))
    out_name = '{}-{}.ppk'.format(t0_code, t1_code)
    fout = open(os.path.join(out_root, out_name),'w')
    print('picking %s'%date)
    data_dict = get_data_dict(date, data_root)
    for data_paths in data_dict.values():
        st  = read(data_paths[0])
        st += read(data_paths[1])
        st += read(data_paths[2])
        picker.pick(st, fout)
    fout.close()

if __name__ == '__main__':
  mp.set_start_method('spawn', force=True) # or 'forkserver'

  # start picking
  start_time, end_time = [UTCDateTime(time) for time in time_range.split('-')]
  num_days = int((end_time - start_time) / 86400)
#  for day_idx in range(num_days): pick_day(start_time+86400*day_idx)
  
  pool = mp.Pool(10)
  for day_idx in range(num_days):
    pool.apply_async(pick_day, args=(start_time+86400*day_idx,))
  pool.close()
  pool.join()
  
