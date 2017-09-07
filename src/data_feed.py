import numpy as np
import pandas as pd
from argparse import ArgumentParser
from progbar import Progbar
# from sklearn.model_selection import KFold
# from os.path import isdir
# from os import listdir
import os


def _strip_names(direc):
    return [mem.rsplit('.')[0] for mem in
            os.listdir(direc)]


class Data_Feed(object):
    """Data_Feed batch generator

    Provides...
    """

    def __init__(self, label_dir, data_dir, region_size=1440,
                 rand_seed=None):
        if not isdir(label_dir):
            raise TypeError("label_dir must be a directory")
        if not isdir(data_dir):
            raise TypeError("data_dir must be a directory")
        if not isinstance(region_size, int):
            raise TypeError("region_size should be an int")
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.region_size = region_size
        self.rand_seed = rand_seed
        if not self._chk_files(self.label_dir, self.data_dir):
            raise ValueError("Not all data files are matched to label files")
        self.df = self._load_all(self.region_size)

    def _chk_files(self, label_dir, data_dir):
        data_files = _strip_names(data_dir)
        label_files = _strip_names(label_dir)
        for data_file in data_files:
            if data_file not in label_files:
                return False
        return True

    def _load_all(self, region_size):


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("label_dir", nargs=1, type=str,
                        help="Directory containing Labels as .npy files")
    parser.add_argument("data_dir", nargs=1, type=str,
                        help="Directory containing traing vids as .npy files")
    parser.add_argument("region_size", nargs=1, type=str,
                        help="Frames per continous region. Generator will cut"
                        " videos into continous regions of this size, and then"
                        " re-order randomly these chunks.")
    parser.add_argument("rand_seed", nargs=1, type=str,
                        help="Random seed ")

    args = parser.parse_args()
    main(args)
