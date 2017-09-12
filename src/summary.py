from __future__ import division
from __future__ import print_function
import sys
import argparse
import os
import cv2
import numpy as np
import pandas as pd


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main(directory, command='class_rep'):
    files = os.listdir(direcotry)
    PATH = os.path.realpath(direcotry)
    OUTFILE = command + '.h5'

    def chk_csv(string): return True if '.csv' in string else False

    def get_path(filename): return os.path.join(PATH, filename)
    filenames = map(get_path, filter(chk_csv, files))
    if OUTFILE in files:
        label_df = pd.read_hdf(os.path.join(PATH, OUTFILE))
    for filename in filenames:
        try:
            file_df = pd.read_csv(filename, index_col=0)
            filename = _strip_filename(filename)
            if 'label_df' not in locals():
                label_df = pd.DataFrame(index=file_df.columns,
                                        columns=(filename,))
            if command == 'class_rep':
                label_df[filename] = calc_cls_rep(file_df)
                del file_df
            elif command == 'rate_dur':
                pass
        except IOError:
            eprint("{} file reading error. Does it exist?".format(filename))
    label_df.to_hdf(get_path(OUTFILE), 'w')


def calc_rate_dur(file_df):
    print("Not implemented")
    return


def calc_cls_rep(file_df):
    """
    For every column, get the sum.
    """
    rtn_df = pd.DataFrame(index=file_df.columns)
    rtn_df[0] = 0
    for col_ind, col in enumerate(file_df.columns):
        vals, cnts = np.unique(file_df[col], return_counts=1)
        rtn_df.iloc[col_ind, 0] = sum(
            [val * cnt for val, cnt in zip(vals, cnts)])
    return rtn_df


def _strip_filename(filename):
    filename = filename.rsplit('/', 1)[1]
    filename = filename.rsplit('.', 1)[0]
    return filename


def _calc_per(arr):
    num = sum(arr)
    return [100.0 * elm / num for elm in arr]


def plt_class_rep(label_df, save_file):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.figure()
    label_df.plot.barh(stacked=True, log=True)
    plt.savefig(save_file)
    plt.close('all')
    return None


def print_uniques(label_df):
    for col in label_df.columns:
        vals, cnts = np.unique(label_df[col], return_counts=1)
        if len(vals) == 1:
            out_vals = "0"
            out_per = [100] + [0] * 2
        elif len(vals) == 2:
            out_vals = "0&1"
            out_per = _calc_per(cnts) + [0]
        else:
            out_vals = "all"
            out_per = _calc_per(cnts)
        print(
            "{:<34} {:<3} {:.5f} {:.5f} {:.5f}".format(
                col, out_vals, out_per[0], out_per[1], out_per[2]))
    return


def show_img(array):
    cv2.namedWindow('image', flags=[cv2.WINDOW_NORMAL, cv2.WINDOW_KEEPRATIO,
                                    cv2.WINDOW_GUI_EXPANDED])
    cv2.waitKey(1)
    cv2.imshow('image', array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # modes = parser.add_mutually_exclusive_group()
    # modes.add_argument("-d", nargs='?', help="Set to directory mode")
    # modes.add_argument("-f", nargs='?', help="Set to file mode")
    parser.add_argument("direcotry", nargs=1, type=str,
                        help="Path to Directory containing"
                        " labels in csv format")
    args = parser.parse_args()
    main(args)
