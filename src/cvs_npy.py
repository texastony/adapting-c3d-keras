from numpy import save
from argparse import ArgumentParser
from progbar import Progbar
from pandas import read_csv


def cvs_npy(filename, output_dir):
    label_df = read_csv(filename, index_col=0)
    filename = filename.rsplit('.', 1)[0] + '.npy'
    filename = filename.rsplit('/', 1)[1]
    return save(output_dir + '/' + filename, label_df.values)


def main(args):
    status = Progbar(len(args.filenames))
    for ind, arg in enumerate(args.filenames):
        status.update(ind + 1, text=arg.rsplit('/', 1)[1])
        cvs_npy(arg, args.output_dir[0])
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='*', type=str,
                        help="Which arrays to process")
    parser.add_argument("output_dir", nargs=1, type=str,
                        help="output directory")
    args = parser.parse_args()
    main(args)
