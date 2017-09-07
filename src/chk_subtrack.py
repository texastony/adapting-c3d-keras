from numpy import load, save, transpose
from argparse import ArgumentParser
from progbar import Progbar


def chck_subract(filename, cube_file, output_dir):
    arr = load(filename)
    mean_cube = load(cube_file)
    mean_cube = transpose(mean_cube, (1, 2, 3, 0))
    arr = - mean_cube
    arr = arr[:, 8:120, 30:142, :]
    filename = filename.rsplit('/', 1)[1]
    return save(output_dir + '/' + filename, arr)


def main(args):
    status = Progbar(len(args.filenames))
    for ind, arg in enumerate(args.filenames):
        status.update(ind, text=arg.rsplit('/', 1)[1])
        chck_subract(arg, args.cube_file[0], args.output_dir[0])
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='*', type=str,
                        help="Which arrays to process")
    parser.add_argument("cube_file", nargs=1, type=str,
                        help="Where the mean cube array is")
    parser.add_argument("output_dir", nargs=1, type=str,
                        help="output directory")
    args = parser.parse_args()
    main(args)
