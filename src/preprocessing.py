from cv2 import VideoCapture, resize, CAP_PROP_FRAME_COUNT, \
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, INTER_AREA
from numpy import save, empty, dtype
from argparse import ArgumentParser
from progbar import Progbar
# import multiprocessing


def videoGetter(filename,
                outdir='out/preprocessed/',
                update_rate=50):
    DEST_IMG_SIZE = (171, 128)
    cap = VideoCapture(filename)
    frameCount = int(cap.get(CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(CAP_PROP_FRAME_HEIGHT))
    status = Progbar(frameCount, text=filename)
    # Note that for some reason, cv2 reads images hieght and then width
    buf = empty(
        (frameCount, DEST_IMG_SIZE[1], DEST_IMG_SIZE[0], 3), dtype('uint8'))
    temp = empty((frameWidth, frameHeight, 3), dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, temp = cap.read()
        buf[fc] = resize(src=temp, dst=buf[fc].shape, dsize=DEST_IMG_SIZE,
                         interpolation=INTER_AREA)
        if fc % update_rate == 0:
            status.update(fc)
        fc += 1
    cap.release()
    del cap, temp
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
