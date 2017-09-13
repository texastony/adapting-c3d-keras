from __future__ import print_function
from sys import exitfunc
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time


OUTDIR = os.path.relpath('out')
LABEL_DIR = os.path.relpath('data/train-labels')
RUN_PATH = os.path.relpath('out/learn/run.p')
CLS_LST = ['micromanipulator', 'phacoemulsifier handpiece',
           'irrigation/aspiration handpiece', 'capsulorhexis cystotome']
num_gpus = 16
num_steps = 200
learning_rate = 0.003
batch_size = 1440
display_step = 10

num_input = 8192
num_classes = len(CLS_LST)


def load_stuff(cmd=['label'], path_run=RUN_PATH, path_labels=LABEL_DIR,
               path_extracted=os.path.join(OUTDIR, 'extracted'),
               cls_lst=CLS_LST):
    label_arr = []
    frame_arr = []
    runs = pickle.load(open(path_run, 'rb'))
    srcs = np.array([elm[0] for elm in runs])
    for src in np.unique(srcs):
        runs_here = [mem[1] for mem in
                     filter(lambda elm: True if elm[0] == src else False,
                            runs)]
        if 'label' in cmd:
            file_df = pd.read_csv(os.path.join(
                LABEL_DIR, src + '.csv'), index_col=0)
        if 'frame' in cmd:
            src_arr = np.load(os.path.join(OUTDIR, src + '.npy'))
        for run in runs_here:
            if 'label' in cmd:
                label_arr.append(
                    file_df.loc[run[0]:run[1], cls_lst].values)
            if 'frame' in cmd:
                frame_arr.append(src_arr[run[0]:run[1]])
    return label_arr
    if ('label' in cmd) and (len(cmd) == 1):
        return np.hstack(label_arr)
    if ('frame' in cmd) and (len(cmd) == 1):
        return np.vstack(frame_arr)
    else:
        return np.vstack(frame_arr), np.vstack(label_arr)


def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)
