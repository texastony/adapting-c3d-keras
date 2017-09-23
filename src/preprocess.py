from __future__ import division
from cv2 import VideoCapture, resize, CAP_PROP_FRAME_COUNT, \
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, INTER_AREA
from numpy import save, empty, dtype
from argparse import ArgumentParser
from progbar import Progbar
# import multiprocessing


def videoGetter(filename,
                outdir='out/prepro/',
                update_rate=50):
    DEST_IMG_SIZE = (270, 480)
    cap = VideoCapture(filename)
    frameCount = int(cap.get(CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(CAP_PROP_FRAME_HEIGHT))
    status = Progbar(frameCount, text=filename)
    # Note that for some reason, cv2 reads images hieght and then width
    buf = empty(
        (frameCount, DEST_IMG_SIZE[1], DEST_IMG_SIZE[0], 3), dtype('int8'))
    raw = empty((frameWidth, frameHeight, 3), dtype('uint8'))
    middle = empty((DEST_IMG_SIZE[1], DEST_IMG_SIZE[0], 3), dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, temp = cap.read()
        middle[:, :, :] = resize(
            raw, None, fx=0.25, fy=0.25, interpolation=INTER_AREA)
        buf[fc] = (middle.astype('int8') - 255 // 2)
        if fc % update_rate == 0:
            status.update(fc)
        fc += 1
    cap.release()
    del cap, raw, middle
    filename = filename.rsplit('/', 1)[1]
    outpath = outdir + filename.rsplit('.', 1)[0] + '.npy'
    return save(outpath, buf)


def main(args):
    for arg in args.filenames:
        videoGetter(arg)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='*', type=str,
                        help="Which videos to process")
    args = parser.parse_args()
    main(args)
