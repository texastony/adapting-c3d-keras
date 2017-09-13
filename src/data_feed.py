from __future__ import division
from __future__ import print_function
from sys import stderr, exitfunc
from os import listdir, curdir
from argparse import ArgumentParser
from os.path import join as pathjoin
from numpy import power
from random import choice
import pandas as pd
import cPickle as pickle


# cls_lst=['phacoemulsifier handpiece']; path=curdir; min_len=50; min_frame_cnt=[2, 3];cls=cls_lst[0]
def main(cls_lst, path, min_len=50, min_frame_cnt=[2, 5]):
    """Generate Samples based on classes in cls_lst

    Args:
        path : the path to the directory with .h5 files
        cls_lst : a list of the classes you are generating samples for
    """
    try:
        runs_file = filter(lambda name: True if 'runs.h5' in name else False,
                           listdir(path))[0]
        lens_file = filter(
            lambda name: True if 'lens.h5' in name else False,
            listdir(path))[0]
    except IndexError:
        eprint("Could not find runs.hf files in {}".format(path))
        return
    min_frame_cnt = min_frame_cnt[0] * power(10, min_frame_cnt[1])
    runs_df = pd.read_hdf(pathjoin(path, runs_file))
    lens_df = pd.read_hdf(pathjoin(path, lens_file))
    run = []
    for cls in cls_lst:
        try:
            frame_cnt = 0
            long_enough = lens_df.loc[cls].apply(lambda lst:
                                                 map(lambda lgn:
                                                     lgn > min_len, lst))
            [long_enough.drop(elm, axis=0, inplace=True)
             if ((long_enough.loc[elm] == []) or
                 (any(long_enough.loc[elm]) is False))
             else None for elm in long_enough.index]
            while frame_cnt < min_frame_cnt:
                smp_pick = choice(list(enumerate(long_enough.values)))
                run_pick = choice(list(enumerate(smp_pick[-1])))
                while not run_pick[-1]:
                    run_pick = choice(list(enumerate(smp_pick[-1])))
                src = long_enough.index.values[smp_pick[0]]
                run.append((src,
                            runs_df.loc[cls, src][run_pick[0]],))
                frame_cnt += lens_df.loc[cls, src][run_pick[0]]
        except IndexError:
            eprint('class {} has problems'.format(cls))
    pickle.dump(run, open("run.p", "wb"))
    return exitfunc()


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
    main(args.cls_lst, args.path[0], args.min_len, args.min_frame_cnt)
# args = parser.parse_args(['--min_len', '50', '--min_frame', '2','3', str(cls_lst), '.',])
