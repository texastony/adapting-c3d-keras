import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.backend import get_session
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

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


def load_model():
    global model
    model_weight_filename = 'models/sports1M_weights_tf.h5'
    model_json_filename = 'models/sports1M_weights_tf.json'
    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())
    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)


def prnt_mdl_outs(model, batch_size=1):
    last_output = (batch_size, 16, 112, 112, 3)
    for ind, layer in enumerate(model.layers):
        last_output = layer.compute_output_shape(last_output)
        try:
            stride = layer.strides
            print("{:<4} {:>11} {:>10} {:>26} {:>2}".format(
                ind, layer.name, stride, last_output, layer.trainable))
        except AttributeError:
            print("{:<4} {:>11} {:>10} {:>26} {:>2}".format(
                ind, layer.name, ''.join([' '] * 10), last_output,
                layer.trainable))


def modify_model(model, output_units=18,
                 optimizer=SGD(lr=0.003, decay=3.33e-6),
                 loss=mean_squared_error):
    """
    Function for modifying C3D into C3D feature extractor and setting up the
    network for fine tuning. Optimizer is as given in the paper. Loss is best
    guess at what they used.

    # Arguments
        model: a keras model, assumed to be C3D
        output_units: the size of the output vector
        optimizer: a keras optimizer, default is SGD as in C3D paper
        loss: a keras loss, default is MSE
    # Returns
        None
    """
    #  Lock trained conv layers
    for layer in model.layers[:14]:
        layer.trainable = False
    #  Change output vector for new use
    model.layers[-1].units = output_units
    #  Reset top Layers
    reinitialize_layer_weigt(model.layers[14:])
    model.compile(optimizer=optimizer,
                  loss=loss)


def reinitialize_layer_weigt(layers):
    """
    Function by YoelShoshan on Github
    https://github.com/fchollet/keras/issues/341
    Copied on 05/09/2017
    """
    session = get_session()
    for layer in layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}'.format(layer.name, v))
