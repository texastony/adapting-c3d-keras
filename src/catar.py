from __future__ import print_function
import os
import time
import cPickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from progbar import Progbar

NUM_INPUT = 8192
OUTDIR = os.path.relpath('out')
LABEL_DIR = os.path.relpath('data/train-labels')
CLS_LST = ['micromanipulator', 'phacoemulsifier handpiece',
           'irrigation/aspiration handpiece', 'capsulorhexis cystotome']


class Catar(object):
    """Modeled after:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    """

    def __init__(self, path_run=OUTDIR,
                 path_extracted=os.path.join(OUTDIR, 'extracted'),
                 cls_lst=CLS_LST):
        print("Loading Train_Frames")
        self.train_frames = load_stuff(['frame'],
                                       path_run=os.path.join(
                                           path_run, 'learn/run.p'))
        self.train_labels = load_stuff(['label'],
                                       path_run=os.path.join(
                                           path_run, 'learn/run.p'))
        print("Loading Test_Frames")
        self.test_frames = load_stuff(['frame'],
                                      path_run=os.path.join(path_run, 'vldt/run.p'))
        self.test_labels = load_stuff(['label'],
                                      path_run=os.path.join(path_run, 'vldt/run.p'))
        self.test_ind = 0
        self.train_ind = 0

    def _next_batch(self, request_size, arr, ind):
        end = len(arr)
        if (ind + request_size) < end:
            return ind + request_size, arr[ind:ind + request_size]
        else:
            rtn_ind = request_size - (end - ind)
            rtns = np.arange(ind - end, rtn_ind, 1)
            return rtn_ind, arr[np.ix_(rtns)]

    def train_next_batch(self, request_size):
        _, frame_arr = self._next_batch(request_size, self.train_frames,
                                        self.train_ind)
        self.train_ind, label_arr = self._next_batch(request_size,
                                                     self.train_labels,
                                                     self.train_ind)
        return frame_arr, label_arr

    def test_next_batch(self, request_size):
        _, frame_arr = self._next_batch(request_size, self.test_frames,
                                        self.test_ind)
        self.test_ind, label_arr = self._next_batch(request_size,
                                                    self.test_labels,
                                                    self.test_ind)
        return frame_arr, label_arr


def load_stuff(cmd=['label'],
               path_run='out/learn/run.p', path_labels='data/train-labels',
               path_extracted=os.path.join(OUTDIR, 'extracted'),
               cls_lst=CLS_LST):
    runs = pickle.load(open(path_run, 'rb'))
    srcs = np.array([elm[0] for elm in runs])
    height = np.sum([elm[1][1] - elm[1][0] for elm in runs])
    width = NUM_INPUT if 'frame' in cmd else len(cls_lst)
    out_arr = np.empty((height, width))
    out_ind = 0
    uni_srcs = np.unique(srcs)
    status = Progbar(height)
    for ind, src in enumerate(uni_srcs):
        runs_here = [mem[1] for mem in
                     filter(lambda elm: True if elm[0] == src else False,
                            runs)]
        if 'label' in cmd:
            file_df = pd.read_csv(os.path.join(
                path_labels, src + '.csv'), index_col=0)
        if 'frame' in cmd:
            src_arr = np.load(os.path.join(path_extracted, src + '.npy'))
        for run in runs_here:
            next_ind = out_ind + (run[1] - run[0])
            temp = out_arr[out_ind:next_ind + 1, :].shape
            if temp[0] == 0:
                return out_arr
            status.update(
                next_ind, text="{}  ".format(temp))
            if 'label' in cmd:
                out_arr[out_ind:next_ind + 1, :] = file_df.loc[
                    run[0]: run[1], cls_lst].values
            if 'frame' in cmd:
                out_arr[out_ind:next_ind + 1, :] = src_arr[run[0]:run[1]]
            out_ind += next_ind
    return out_arr


def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)


os.chdir('..')
load_stuff()
