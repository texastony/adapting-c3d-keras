import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from keras.backend import get_session
from keras.optimizers import SGD
from keras.metrics import mean_squared_error
from keras.models import load_model
from keras import backend as K
from progbar import Progbar


os.chdir('/home/ubuntu/capstone')
OUTDIR = os.path.relpath('out')
DATADIR = os.path.relpath('data/preprocessed')
MODEL_DIR = os.path.relpath('model')
UPDT_FRQ = 50
# main(['test00.npy'])

def main(filenames, device='/gpu:2'):
    print("[Info] Loading Model")
    sess = tf.Session()
    K.set_session(sess)
    model = load_model(os.path.join(MODEL_DIR, 'from_tony.h5'))
    to_tf = tf.placeholder(tf.float32, shape=(None, 16, 112, 112, 3))
    output_tensor = model(to_tf)
    mean_cube = np.load(os.path.join(MODEL_DIR, 'train01_16_128_171_mean.npy'))
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
    to_net = np.empty((16, 112, 112, 3))
    for filename in filenames:
        print("[Info] Loading videos")
        vid = np.load(os.path.join(OUTDIR, 'preprocessed', filename)).astype(np.float32)
        temp = np.zeros((16,)+vid.shape[1:])
        end_frame = vid.shape[0]
        output = np.empty((vid.shape[0], 8192))
        status = Progbar(target=end_frame, text=filename.rsplit('.', 1)[0])
        for frm_ind, frame in enumerate(vid):
            if frm_ind < 8:
                temp[(8 - frm_ind):] = vid[:frm_ind + 8]
                temp[:(8 - frm_ind)] = 0
            elif end_frame < frm_ind + 8:
                temp[:(16 + (end_frame - (frm_ind + 8)))] = vid[frm_ind - 8:]
            else:
                temp[:, :, :, :] = vid[(frm_ind - 8):(frm_ind + 8), :, :, :]
            temp -= mean_cube
            to_net[:, :, :, :] = temp[:, 8:120, 30:142, :]
            output[frm_ind] = sess.run(output_tensor,
                                       feed_dict={to_tf: np.array([to_net])})
            # with tf.device(device):
            #     output[frm_ind] = model.predict_on_batch(np.array([to_net]))
            if frm_ind % 50:
                status.update(frm_ind)
        np.save(os.path.join(OUTDIR, 'extracted', filename), output)

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
    # reinitialize_layer_weigt(model.layers[14:])
    model.layers = model.layers[:15]
    model.compile(optimizer=optimizer,
                  loss=loss)
    return model


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
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='*', type=str,
                        help="Which videos to process")
    # parser.add_argument("device", nargs=1, type=str,
    #                     help="Which gpu to use")
    args = parser.parse_args()
    main(args.filenames)
