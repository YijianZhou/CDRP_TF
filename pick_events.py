""" Repick Gorkha events
"""
import sys, os, glob, shutil
sys.path.append('/home/zhouyj/software/CDRP_TF/')
sys.path.append('/home/zhouyj/software/data_prep')
import time
from obspy import read
import multiprocessing as mp
import picker_event as picker
from signal_lib import preprocess
import warnings
warnings.filterwarnings("ignore")

# i/o paths
event_root = '/data3/bigdata/zhouyj/Gorkha_events'
stz_paths = glob.glob(os.path.join(event_root, '*/*.*HZ.sac'))
ckpt_dir = '/home/zhouyj/Gorkha/run_cdrp/output/rc_ppk/PpkNet'
fout = open('output/xq_pad_ai.repick','w')
cdrp_dir = '/home/zhouyj/software/CDRP_TF'
shutil.copyfile('config_gorkha.py', os.path.join(cdrp_dir,'seisnet','config.py'))
# params
to_prep = True
samp_rate = 100
freq_band = ['bandpass',[1.,40.]]
num_workers = 10
batch_size = 5000
ckpt_step = 9500
picker = picker.CDRP_Picker_Event(ckpt_dir=ckpt_dir, ckpt_step=ckpt_step)
event_win = [5,35] # to event window

def read_stream(stz_path):
    # get path
    event_dir, fname = os.path.split(stz_path)
    event_name = os.path.basename(event_dir)
    net, sta, chn, _ = fname.split('.')
    event_sta = ','.join([event_name, sta])
    fnames = '.'.join([net, sta, '*'])
    st_paths = sorted(glob.glob(os.path.join(event_dir, fnames)))
    # read data
    stream  = read(st_paths[0])
    stream += read(st_paths[1])
    stream += read(st_paths[2])
    start_time = stream[0].stats.starttime
    return stream.slice(start_time+event_win[0], start_time+event_win[1]), event_sta


def write_pick(picks, name_list, fout):
    num_samp = len(picks)
    for i in range(num_samp):
        tp, ts = picks[i]
        event_sta = name_list[i]
        fout.write('{},{},{}\n'.format(event_sta, tp, ts))


num_samp = len(stz_paths)
num_batch = 1 + num_samp // batch_size
ppk_time = 0
t=time.time()
print('%s batches'%num_batch)
for batch_idx in range(num_batch):
    print('-'*40)
    print('picking {}th batch | {:.2f}s'.format(batch_idx+1, time.time()-t))

    # read data
    print('read data')
    pool = mp.Pool(processes=num_workers)
    outs = pool.map_async(read_stream, stz_paths[batch_idx*batch_size : (batch_idx+1)*batch_size])
    pool.close()
    pool.join()
    streams = [out[0] for out in outs.get()]
    if to_prep: streams = [preprocess(out[0], samp_rate, freq_band) for out in outs.get()]
    name_list = [out[1] for out in outs.get()]

    # run CDRP
    picks = picker.pick(streams)
    write_pick(picks, name_list, fout)

fout.close()
