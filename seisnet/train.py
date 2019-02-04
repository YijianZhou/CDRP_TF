"""
Train CDRP model
"""
import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/CDRP_TF')
import numpy as np
import tensorflow as tf
# import model
import seisnet.config as config
import seisnet.data_pipeline as dp
import seisnet.models as models
from tflib.nn_model import BaseModel

def get_det_samples(dataset_class, data_shape):

    pos_dir = os.path.join(args.data_dir, dataset_class, 'positive')
    neg_dir = os.path.join(args.data_dir, dataset_class, 'negative')
    pos_feeder = dp.Feeder(pos_dir, data_shape)
    neg_feeder = dp.Feeder(neg_dir, data_shape)
    pos_samples = {'data': pos_feeder.data,
                  'label': pos_feeder.det_label}
    neg_samples = {'data': neg_feeder.data,
                  'label': neg_feeder.det_label}
    samples = {
        'data':  tf.concat([pos_samples["data"],  neg_samples["data"]],  axis=0),
        'label': tf.concat([pos_samples["label"], neg_samples["label"]], axis=0)
        }

    return samples


def get_ppk_samples(dataset_class, data_shape):

    seq_dir = os.path.join(args.data_dir, dataset_class, 'sequence')
    ppk_feeder = dp.Feeder(seq_dir, data_shape)
    samples = {
        'data':  ppk_feeder.data,
        'label': ppk_feeder.ppk_label
        }

    return samples


def main(args):

    # setup training
    cfg = config.Config()
    win_points_len = 100 *int(cfg.win_len)
    if   args.model=='DetNet':
        num_steps = 1
        data_shape = [cfg.cnn_bsize, num_steps, win_points_len, cfg.num_chns]
    elif args.model=='PpkNet':
        step_len    = int(100*cfg.step_len)
        step_stride = int(100*cfg.step_stride)
        num_steps   = -(step_len/step_stride-1) + win_points_len/step_stride
        data_shape = [cfg.rnn_bsize, num_steps, step_len, cfg.num_chns]
    else: print 'false model name!'

    # get training and validation set
    if   args.model=='DetNet':
        train_samples = get_det_samples('train', data_shape)
        valid_samples = get_det_samples('valid', data_shape)
    elif args.model=='PpkNet':
        train_samples = get_ppk_samples('train', data_shape)
        valid_samples = get_ppk_samples('valid', data_shape)
    inputs = [train_samples, valid_samples]

    # get model
    ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if   args.model=='DetNet':
        model = models.DetNet(inputs, ckpt_dir)
    elif args.model=='PpkNet':
        model = models.PpkNet(inputs, ckpt_dir)
    # train
    BaseModel(model).train(args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help = 'which model to train')
    parser.add_argument('--data_dir', type=str,
                        default='/home/zhouyj/Documents/CDRP_TF/data/tmp')
    parser.add_argument('--ckpt_dir', type=str,
                        default='/home/zhouyj/Documents/CDRP_TF/output/tmp')
    parser.add_argument('--resume', default=False)
    args = parser.parse_args()
    main(args)

