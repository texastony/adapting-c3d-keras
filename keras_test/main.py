import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
import cv2
import c3d_model
import sys

OUTDIR = os.path.relpath('../out/')
DATADIR = os.path.relpath('../data')


def main():
    model_weight_filename = 'models/sports1M_weights_tf.h5'
    model_json_filename = 'models/sports1M_weights_tf.json'

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)

    print("[Info] Loading labels...")
    labels_df = pd.read_csv('../train_labels/train01.csv',  index_col=0)

    X = videoGetter('../data/train01.mp4', False, True).astype(np.float32)
    X = X[140:140 + 16]
    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    X -= mean_cube
    X = X[:, 8:120, 30:142, :]
    output = model.predict_on_batch(np.array([X]))
    print(output)


def videoGetter(filename, try_save=False, try_reload=True):
    DEST_IMG_SIZE = (128, 171)
    if (try_reload or try_save):
        chk_pth = filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
        chk_pth = OUTDIR + "/preprocessed/{}.npy".format(chk_pth)
        if os.path.isfile(chk_pth) and try_reload:
            print("Found Preprocessed")
            return np.load(chk_pth)

    print("[Info] Reading Video")
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, DEST_IMG_SIZE[0], DEST_IMG_SIZE[1],
                    3), np.dtype('uint8'))
    temp = np.empty((frameHeight, frameWidth, 3),
                    np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, temp = cap.read()
        cv2.resize(src=temp, dst=buf[fc], dsize=DEST_IMG_SIZE,
                   interpolation=cv2.INTER_AREA)
        fc += 1
    cap.release()

    if try_save:
        np.save(chk_pth, buf)

    return buf
