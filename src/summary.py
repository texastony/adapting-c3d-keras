import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2


def main(args):
    for file_name in args.file_names:
        label_df = pd.read_csv(file_name, index_col=0)
        timeline = label_df.sum(axis=1)
        if args.verbose:
            print("Full Postive Count: {} \nFull Negative Count: {} \n"
                  "Mixed Count: {}".format(len(timeline[timeline >= 1]),
                                           len(timeline[timeline == 0]),
                                           len(timeline[timeline == 0.5])))
        file_name = file_name.rsplit('.', 1)[0] + '.npy'
        np.save(file_name, timeline.values)


def show_img(array):
    cv2.namedWindow('image', flags=[cv2.WINDOW_NORMAL, cv2.WINDOW_KEEPRATIO,
                                    cv2.WINDOW_GUI_EXPANDED])
    cv2.waitKey(1)
    cv2.imshow('image', array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def test(args):
    print(args.file_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_names", nargs='*', type=str,
                        help="Files to process and save simplified reps")
    parser.add_argument("-v", '--verbose', action='store_true',
                        help='increase output verbosity')
    args = parser.parse_args()
    main(args)
