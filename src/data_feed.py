from __future__ import division
from __future__ import print_function
from sys import stderr
from os import listdir, curdir
from argparse import ArgumentParser
from os.path import join as pathjoin
from numpy import power
from random import choice
import pandas as pd
from progbar import Progbar

# CLS_LST = ['micromanipulator', 'phacoemulsifier handpiece', 'irrigation/aspiration handpiece', 'capsulorhexis cystotome']
# PATH = '../out/learn/'
# from data_feed import *; obj = DataFeed(CLS_LST, PATH); run = obj._main()
# sum(run.lens.sum())


class DataFeed(object):

    def __init__(self, cls_lst, path, min_len=50, min_frame_cnt=[2, 5]):
        self.cls_lst = cls_lst
        self.path = path
        self.min_len = min_len
        self.min_frame_cnt = min_frame_cnt[0] * power(10, min_frame_cnt[1])
        self.runs_file, self.lens_file = self._get_files()
        self.runs_df = pd.read_hdf(pathjoin(path, self.runs_file))
        self.lens_df = pd.read_hdf(pathjoin(path, self.lens_file))
        self.run = []
        self.long_enough = pd.DataFrame()

    def _exit(self):
        out = pd.Series(index=[elm[0] for elm in self.run],
                        data=[elm[1] for elm in self.run])
        temp = pd.DataFrame(index=pd.unique(out.index),
                            data={'runs': [pd.unique(out[elm])
                                           for elm in pd.unique(out.index)]})
        temp['lens'] = [[elm[1] - elm[0] for elm in mem]
                        for mem in temp['runs'].values]
        temp.to_hdf(pathjoin(self.path, 'run.h5'), 'w')
        print("Total Frame Count: {}".format(sum(temp.lens.sum())))
        return

    def _get_files(self):
        try:
            runs_file = filter(lambda name:
                               True if 'runs.h5' in name else False,
                               listdir(self.path))[0]
            lens_file = filter(lambda name:
                               True if 'lens.h5' in name else False,
                               listdir(self.path))[0]
        except IndexError:
            eprint("Could not find runs.hf files in {}".format(self.path))
        return runs_file, lens_file

    def chk_long_enough(self):
        [self.long_enough.drop(elm, axis=0, inplace=True)
         if ((self.long_enough.loc[elm] == []) or
             (any(self.long_enough.loc[elm]) is False))
         else None for elm in self.long_enough.index]

    def _main(self):
        """Generate Samples based on classes in cls_lst
        Args:
            path : the path to the directory with .h5 files
            cls_lst : a list of the classes you are generating samples for
        """
        for cls in self.cls_lst:
            try:
                self._work(cls)
            except IndexError:
                eprint('class {} has problems'.format(cls))
        print("\n")
        return self._exit() if len(self.run) > 0 else eprint("Disaster")

    def _work(self, cls):
        frame_cnt = 0
        self.long_enough = self.lens_df.loc[cls].apply(
            lambda lst: map(lambda lgn: lgn > self.min_len, lst))
        status = Progbar(self.min_frame_cnt, text=cls)
        self.chk_long_enough()
        while frame_cnt < self.min_frame_cnt:
            self.chk_long_enough()
            if len(self.long_enough.index) == 0:
                status.update(frame_cnt, force=True)
                print("\n")
                return
            smp_pick = choice(list(enumerate(self.long_enough.values)))
            run_pick = choice(list(enumerate(smp_pick[-1])))
            while not run_pick[-1]:
                run_pick = choice(list(enumerate(smp_pick[-1])))
            src = self.long_enough.index.values[smp_pick[0]]
            self.run.append((src,
                             self.runs_df.loc[cls, src]
                             [run_pick[0]],))
            frame_cnt += self.lens_df.loc[cls, src][run_pick[0]]
            status.update(frame_cnt)
            self.long_enough.iloc[smp_pick[0]][run_pick[0]] = False
            self.chk_long_enough()


def _strip_all(filename):
    return int(filename.rsplit('.')[0].split('train')[-1])


def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def _strip_names(direc):
    return [mem.rsplit('.')[0] for mem in listdir(direc)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cls_lst", nargs='*', type=str,
                        help="list of classes to select samples for."
                        " NOTE: Nothing will always be included")
    parser.add_argument("path", nargs=1, type=str,
                        default=curdir,
                        help="path to Directory with runs.h5 and lens.h5")
    parser.add_argument("--min_len", nargs=1, type=int,
                        default=50,
                        help="minimum length for a run")
    parser.add_argument("--min_frame_cnt", nargs=2, type=int,
                        default=[2, 3],
                        help="minimum number of frames per class")
    args = parser.parse_args()
    DataFeed(args.cls_lst, args.path[0],
             args.min_len, args.min_frame_cnt)._main()
# examp_arg = ['--min_len', '50', '--min_frame', '2','3', str(cls_lst), '.',]
# parser.parse_args()
