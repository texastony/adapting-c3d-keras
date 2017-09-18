from __future__ import print_function
import os
import time
import multi_gpu
import numpy as np
import pandas as pd
import tensorflow as tf
from progbar import Progbar
from keras import layers, optimizers, metrics, activations, losses
from keras.models import Sequential, load_model, model_from_json
from collections import OrderedDict
from catar import *

NUM_GPUS = 16
NUM_INPUT = 8192
NUM_CLASSES = 21  # len(catar.CLS_LST)
DROPOUT = 0.5
pthjn = os.path.join


def build_net(input_shape, n_classes, dropout):
    """Creates a dense multi-layer network"""
    model = Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation='sigmoid'))
    # opt = optimizers.Nadam(lr=0.003)
    # los = losses.binary_crossentropy
    # met = [metrics.binary_accuracy, metrics.mean_squared_error]
    # # model = make_parallel(model, NUM_GPUS)
    # model.compile(loss=los, optimizer=opt, metrics=met)
    return model


def build_from_I3(input_shape, n_classes, dropout):
    """Creates a dense multi-layer network"""
    model = Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation=igmo))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# def main():
os.chdir('/home/ubuntu/capstone/')
print('Loading train-labels\n')
label_dict = load_dfs('data/train-labels')
labels = join_dfs(label_dict, 21)
print('Loading train-frames')
frames = get_data(label_dict, 'out/extracted/train/')
print('\nLoading validation labels')
vldt_dict = load_dfs('data/test-labels')
vldt_labels = join_dfs(vldt_dict, 21)
print('Loading validation Frames')
vldt_frames = get_data(vldt_dict, 'out/extracted/test/')
# print('\nInit Model')
# model = build_net(frames.shape[1:], 21, DROPOUT)
# # # model.load_weights('model/top_weights.h5')
# opt = optimizers.Nadam(lr=0.003)
# los = losses.binary_crossentropy
# met = [metrics.binary_accuracy, metrics.mean_squared_error]
# print('Parallizing model')
# model = multi_gpu.make_parallel(model, NUM_GPUS)
# model.compile(loss=los, optimizer=opt, metrics=met)
# train_size = frames.shape[0] / NUM_GPUS
# train_limit = (train_size * NUM_GPUS) - frames.shape[0]
# vldt_size = vldt_frames.shape[0] / NUM_GPUS
# vldt_limit = (vldt_size * NUM_GPUS) - vldt_frames.shape[0]
# tot_epoch = 0
ser_sum = pd.Series(data=np.sum(labels.values, axis=1),
                    index=labels.index.values)
ser_max = pd.Series(data=np.argmax(labels.values, axis=1),
                    index=labels.index.values)
tool_labels = labels[CLS_LST][ser_sum > 0]
tool_frames = frames[tool_labels.index]
train_size = tool_frames.shape[0] / NUM_GPUS
train_limit = (train_size * NUM_GPUS) - tool_frames.shape[0]
# metric_dict = OrderedDict()
# # while tot_epoch < 20:
model.fit(x=tool_frames[:train_limit], y=tool_labels.values[:train_limit],
          batch_size=train_size, epochs=1, initial_epoch=0)
# # print('Evaluation')
# # metric_dict[tot_epoch] = model.evaluate(x=vldt_frames[:vldt_limit],
# # y = vldt_labels.values[:vldt_limit],
# # batch_size = vldt_size)
# # tot_epoch += 5
# # model.save_weights('model/top_weights.h5')
# # new_arr = model.predict(batch_size=vldt_size, x=vldt_frames)


# main()


def prnt_mdl_outs(model, batch_size=1):
    for ind, layer in enumerate(model.layers):
        print("{:<4} {:>11} {}".format(
            ind, layer.name, type(layer)))

# def chk_len(data_path, label_path):
#     labels = load_dfs(label_path)
#     print([df.values.shape[0] for df in labels.values()]
#           == [arr.shape[0] for arr in data.values()])
#     print(labels.keys())
#     print(data.keys())
