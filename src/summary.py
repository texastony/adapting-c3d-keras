from __future__ import division
from __future__ import print_function
from sys import stderr
from argparse import ArgumentParser
from os import listdir, curdir
from os.path import realpath
from os.path import join as pathjoin
import numpy as np
import pandas as pd


def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def main(filenames, path, command='class_rep', classes=[],
         outname=None, rtn_lens=False, rtn_runs=True):
    PATH = realpath(path)
    OUTFILE = command + '.h5'
    if outname is not None:
        if rtn_lens:
            outname += '_rtn_lens_'
        OUTFILE = outname + '_' + OUTFILE

    def chk_csv(string): return True if '.csv' in string else False

    def get_path(filename): return pathjoin(PATH, filename)
    filenames = filter(chk_csv, filenames)
    if (OUTFILE in listdir(PATH)) and (outname is None):
        label_df = pd.read_hdf(pathjoin(PATH, OUTFILE))
    for filename in filenames:
        try:
            file_df = pd.read_csv(get_path(filename), index_col=0)
        except IOError:
            eprint("{} file reading error. Does it exist?".format(filename))
            continue
        filename = _strip_filename(filename)
        if 'label_df' not in locals():
            ind = np.hstack((file_df.columns.values, np.array(['Nothing'])))
            label_df = pd.DataFrame(index=ind,
                                    columns=[filename])
        if 'class_rep' in command:
            label_df[filename] = calc_cls_rep(file_df)
        if 'rate_dur' in command:
            label_df[filename] = calc_rate_dur(
                file_df, file_df.columns.values,
                rtn_lens=rtn_lens, rtn_runs=rtn_runs)
        del file_df
    if command == 'class_rep':
        plt_class_rep(label_df, save_file=get_path(
            outname + '_' + command + '.png'))
    label_df.to_hdf(get_path(OUTFILE), 'w')
    return label_df


def find_runs(df, col, target=1.0):
    """Returns runs of values

    Args:
        df : A pd dataframe
        col : A column in the dataframe

    Returns:
        rtn_df : A pd dataframe, indexed by the unique values
            in col, with columns runs and lens.
            runs has the head and tail of the runs
            lens has the length of the runs
    """
    if col == 'Nothing':
        df['sum'] = df.sum(axis=1)
        col = 'sum'

    def get_head_tail(arr): return (arr[0], arr[-1])

    def get_lens(arr): return len(arr)
    # from: https://stackoverflow.com/a/14360423/2090045
    df['block'] = (df[col].shift(1) != df[col]).astype(int).cumsum()
    out = df.reset_index().groupby([col, 'block'])[
        df.index.name].apply(np.array)
    df.drop('block', axis=1, inplace=True)
    if col == 'sum':
        df.drop('sum', axis=1, inplace=True)
        target = 0.0
    if target not in out.index.levels[0]:
        return [], []
    temp = out.loc[target, :].values
    return map(get_head_tail, temp), map(get_lens, temp)


def calc_rate_dur(file_df, cols, rtn_runs=True, rtn_lens=False):
    """Returns the head and tails of runs and their duration

    Args:
        file_df : pd DataFrame
        cols : columns in file_df to look at

    Returns:
        pd.DataFrame : indexed by cols, with columns runs and lens
    """
    runs = []
    lens = []

    def appender(tup): runs.append(tup[0]), lens.append(tup[1])
    ind = np.hstack((file_df.columns.values, np.array(['Nothing'])))
    rtn_df = pd.DataFrame(index=ind)
    [appender(tup) for tup in
        [find_runs(file_df, col) for col in ind]]
    if rtn_runs:
        rtn_df['runs'] = runs
    if rtn_lens:
        rtn_df['lens'] = lens
    return rtn_df


def calc_cls_rep(file_df):
    """
    For every column, get the sum.
    Args:
        file_df : A pandas DataFrame create from a labels file
    Returns:
        rtn_df : A pandas DataFrame with one column which is the sum of
                 every class in the file
    """
    ind = np.hstack((file_df.columns, np.array(['Nothing'])))
    rtn_df = pd.DataFrame(index=ind)
    rtn_df[0] = 0
    for col_ind, col in enumerate(file_df.columns):
        rtn_df.iloc[col_ind, 0] = file_df[col].sum()
    nothing_test = np.zeros((1, len(file_df.columns)))
    rtn_df.iloc[-1, 0] = sum([
        True if (nothing_test == row).all()
        else False for row in file_df.values])
    return rtn_df


def _strip_filename(filename):
    # filename = filename.rsplit('/', 1)[1]
    filename = filename.rsplit('.', 1)[0]
    return filename


def _calc_per(arr):
    num = sum(arr)
    return [100.0 * elm / num for elm in arr]


def plt_class_rep(label_df, save_file='class_rep.png'):
    """Plots class representation in each file.

    If there are more than 20 files, then hatches will be added
    to all classes over 20
    Args:
        label_df: A pandas dataframe with columns as the label files,
                indexed by the classes
        save_file: The file to save the image to

    Color stacking trick From:
    https://stackoverflow.com/a/31052741/2090045
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.colors as mcolors
    plt.style.use('ggplot')

    def _all_clrs(cmap): return cmap(np.arange(cmap.N))
    colors = np.vstack(map(_all_clrs, [get_cmap('tab20'),
                                       get_cmap('Set1')]))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    fig, ax = plt.subplots(figsize=(20, 14))
    label_df.plot.barh(stacked=True, log=False, ax=ax, cmap=mymap)
    ax.set_title('Class Representation in samples')
    plt.savefig(save_file)
    plt.close('all')


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", nargs=1, type=str, default='class_rep',
                        help="Should be one of: class_rep | rate_dur."
                        "class_rep: calculate how class frequency in label"
                        "files rate_dur: calculate classes intervals in"
                        "label files")
    parser.add_argument("--directory", nargs='?', type=str, default=curdir,
                        const=curdir,
                        help="Path to Directory containing"
                        " labels in csv format to work on all csv present")
    parser.add_argument("--classes", nargs='?', default=[], type=str,
                        const=[],
                        help="Classes to examine, eg biomarker. "
                        "Defaults to all")
    parser.add_argument("--files", nargs='?', type=str, default=None,
                        const=None,
                        help="files to analyze. Will still scan for"
                        " pre-existing summary file")
    parser.add_argument("--outname", nargs='?', type=str, default=None,
                        const=None,
                        help="prefix for output files")
    parser.add_argument("--lens",
                        action='store_true',
                        help="Used with rate_dur to find the lengths"
                        " instead of the head and tails of runs.")
    args = parser.parse_args()
    if args.files is not None:
        filenames = args.files.split(',')
    else:
        try:
            filenames = listdir(args.directory)
        except IOError:
            eprint("Parsing Directory raised an error")
    print(args)
    if args.lens is not None:
        main(filenames, args.directory, args.command[0], args.classes,
             args.outname, rtn_lens=True, rtn_runs=False)
    main(filenames, args.directory,
         args.command[0], args.classes, args.outname)
