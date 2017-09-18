from __future__ import print_function
import os
import numpy as np
import pandas as pd
from progbar import Progbar
from collections import OrderedDict
pthjn = os.path.join


def prog_update(status, cur, text=None):
    status.update(cur, text)


def fill_arr(ind, rtn_arr, arr, lens, end):
    rtn_arr[sum(lens[:ind]):sum(lens[:ind]) +
            end] = arr.astype(np.float32)


def get_data(label_dict, path):
    lens = [arr.values.shape[0] for arr in label_dict.values()]
    rtn_arr = np.empty((sum(lens), 128, 171, 3), dtype=np.float32)
    status = Progbar(len(lens))
    [(fill_arr(
        ind, rtn_arr, np.load(
            pthjn(path, arr.rsplit('.', 1)[0] + '.npy')),
        lens, lens[ind]),
      status.update(ind, text=arr.rsplit('.', 1)[0]),)
     for ind, arr in enumerate(label_dict.keys())]
    return rtn_arr


def load_dfs(path):
    return OrderedDict([(name, pd.read_csv(pthjn(path, name), index_col=0))
                        for name in sorted(os.listdir(path))])


def join_dfs(label_dict, n_classes):
    lens = [arr.values.shape[0] for arr in label_dict.values()]
    rtn_arr = np.empty((sum(lens), n_classes), dtype=np.float32)
    [fill_arr(ind, rtn_arr, df.values, lens, lens[ind])
     for ind, df in enumerate(label_dict.values())]
    return pd.DataFrame(columns=label_dict.values()[0].columns, data=rtn_arr)


class Catar(object):
    """Modeled after:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    """

    def __init__(self, label_arr, frame_arr, num_gpus=1):
        self.num_gpus = num_gpus
        self.label_arr = label_arr
        self.frame_arr = frame_arr
        self.batch_szie = frame_arr.shape[0] / num_gpus
        self.index = 0

    def _next_batch(self, request_size, arr, ind):
        end = len(arr)
        if (ind + request_size) < end:
            return ind + request_size, arr[ind:ind + request_size]
        else:
            rtn_ind = request_size - (end - ind)
            rtns = np.arange(ind - end, rtn_ind, 1)
            return rtn_ind, arr[np.ix_(rtns)]

    def next_batch(self, request_size):
        _, frame_arr = self._next_batch(
            request_size, self.frame_arr, self.index)
        self.index, label_arr = self._next_batch(
            request_size, self.label_arr, self.index)
        return frame_arr, label_arr

    def generate(self):
        while True:
            frame_arr, label_arr = self.next_batch(self.batch_size)
            yield frame_arr, label_arr

    # def test_next_batch(self, request_size):
    #     _, frame_arr = self._next_batch(
    #         request_size, self.test_frames, self.test_ind)
    #     self.test_ind, label_arr = self._next_batch(
    #         request_size, self.test_labels, self.test_ind)
    #     return frame_arr, label_arr
