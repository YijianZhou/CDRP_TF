"""
Make training and validation set (in TFRecords)
stream format: [aug_num].[idx].miniseed
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/software/CDRP_TF')
import argparse, glob
import numpy as np
from obspy import read
import multiprocessing as mp
# import model
import seisnet.data_pipeline as dp
import seisnet.config as config
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/data2/WC_train/test_stead_60s')
    parser.add_argument('--out_dir',  type=str,
                        default='/data2/WC_train/tmp')
    parser.add_argument('--sample_class', type=str,
                        help='events, noise or ppk')
    parser.add_argument('--dataset_class', type=str,
                        help='train or validation')
    args = parser.parse_args()


def preprocess(stream):
    stream = stream.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=5.)
    stream = stream.filter('highpass', freq=1.)
    return stream.normalize()


def stream2steps(st_data, num_steps, step_len, step_stride):
    """ convert stream: num_chns * win_len
    to time steps: num_steps * step_len * num_chns
    """
    num_steps   = int(num_steps)
    step_len    = int(step_len)
    step_stride = int(step_stride)

    # make time steps
    num_chns = st_data.shape[0]
    time_steps = np.zeros((num_steps, num_chns, step_len), dtype=np.float32)
    # convert to time steps
    for i in range(num_steps):
        idx0 = i *step_stride
        idx1 = idx0 + step_len
        current_step = st_data[:, idx0:idx1]
        time_steps[i, :, :] = current_step
    return np.transpose(time_steps, [0,2,1])


def mk_ppk_label(tp, ts, num_steps, step_len, step_stride):

    if  tp<step_len: 
        idx_p=0
    else:
        idx_p = int((tp-step_len)/step_stride) +1
    idx_s = int((ts-step_len)/step_stride) +1
    if  idx_s>num_steps: idx_s = num_steps

    # label: 'Noise', 'P wave' and 'S wave'
    label_n =   np.zeros(idx_p)
    label_p =   np.ones(idx_s - idx_p)
    label_s = 2*np.ones(num_steps - idx_s)

    label = np.concatenate([label_n, label_p, label_s])
    return label


def get_class():

    # mk dataset_class
    if  args.dataset_class in ['train', 'training', 'Train', 'Training']:
        dset_class = 'train'
    if  args.dataset_class in ['validation', 'Validation', 'valid']:
        dset_class = 'valid'

    # make sample_class
    if  args.sample_class in ['events', 'event', 'Events', 'Event']:
        samp_class = 'positive'
        det_label  = 1
    if  args.sample_class in ['noise', 'Noise', 'negative']:
        samp_class = 'negative'
        det_label  = 0
    if  args.sample_class in ['ppk', 'frames', 'time_steps', 'sequence']:
        samp_class = 'sequence'
        det_label  = 1

    return dset_class, samp_class, det_label


# write TFRrd file
def write_tfr(out_path):
    # define TFR writer
    writer = dp.Writer(out_path)
    stream_paths = tfr_dict[out_path]

    for stream_path in stream_paths:

        # get stream info
        fdir, fname = os.path.split(stream_path)
        aug_idx, samp_idx, chn = fname.split('.')

        # read stream
        st_paths = sorted(glob.glob(os.path.join(fdir, '{}.{}.*'\
                   .format(aug_idx, samp_idx))))
        if len(st_paths)!=3: print('missing trace!'); continue
        st  = read(st_paths[0])
        st += read(st_paths[1])
        st += read(st_paths[2])

        # drop bad data & preprocess
        if  0. in st.max(): print('brocken trace!'); continue
        if  len(st[0].data) - win_len not in [-1,0,1] or\
            len(st[1].data) - win_len not in [-1,0,1] or\
            len(st[2].data) - win_len not in [-1,0,1]:
            print('missing data points!'); continue
        st = preprocess(st)

        # make data
        xdata = np.float32(st[0].data)
        ydata = np.float32(st[1].data)
        zdata = np.float32(st[2].data)
        st_data = np.array([xdata, ydata, zdata])
        # to time steps
        if samp_class=='sequence':
            time_steps = stream2steps(st_data, num_steps, step_len, step_stride)
        else:
            time_steps = stream2steps(st_data, 1, win_len, win_len)

        # make label
        tp = st[0].stats.sac.t0
        ts = st[0].stats.sac.t1
        if samp_class=='sequence':
            ppk_label = mk_ppk_label(tp*100, ts*100, num_steps, step_len, step_stride)
        else:
            ppk_label = mk_ppk_label(0, 0, 1, win_len, win_len)

        # Write tfrecords
        writer.write(time_steps, det_label, ppk_label)
        print("Making TFRecord {}.{} samples, {}th aug, idx = {}"\
        .format(samp_class, dset_class, aug_idx, samp_idx))
    writer.close()


# setup configure
cfg = config.Config()
win_len     = cfg.win_len *100
step_len    = cfg.step_len *100
step_stride = cfg.step_stride *100
num_steps   = int(-(step_len/step_stride-1) + win_len/step_stride)

# config class --> out_dir
dset_class, samp_class, det_label = get_class()
out_dir = os.path.join(args.out_dir, dset_class, samp_class)
if not os.path.exists(out_dir): os.makedirs(out_dir)
stream_paths = glob.glob(os.path.join(args.data_dir, 
                 dset_class, ['negative','positive'][det_label], '*.*HZ'))

# collect TFR files
tfr_dict = {}
for stream_path in stream_paths:

    # get stream info
    fdir, fname = os.path.split(stream_path) 
    aug_idx, samp_idx, chn = fname.split('.')
    arch_idx = int(samp_idx)//2000

    # set TFR file path
    out_name = '{}_{}_{}_{}.tfrecords'\
               .format(dset_class, samp_class, aug_idx, 2000*arch_idx)
    out_path = os.path.join(out_dir, out_name)
    if out_path not in tfr_dict: tfr_dict[out_path] = [stream_path]
    else: tfr_dict[out_path].append(stream_path)


# write TFRrd file
pool = mp.Pool(processes=10)
pool.map(write_tfr, list(tfr_dict.keys()))
pool.close()
pool.join()
