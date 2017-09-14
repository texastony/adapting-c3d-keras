from __future__ import print_function
from sys import exitfunc
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time


OUTDIR = os.path.relpath('out')
LABEL_DIR = os.path.relpath('data/train-labels')
RUN_PATH = os.path.relpath('out/learn/run.p')
CLS_LST = ['micromanipulator', 'phacoemulsifier handpiece',
           'irrigation/aspiration handpiece', 'capsulorhexis cystotome']
NUM_GPUS = 16
NUM_STEPS = 200
LEARNING_RATE = 0.003
BATCH_SIZE = 1440
DISPALY_STEP = 10

NUM_INPUT = 8192
NUM_CLASSES = len(CLS_LST)
DROPOUT = 0.75


def build_net(x_in, n_classes, dropout, reuse, is_training):
    """Creates a dense multi-layer network"""
    # Define a scope for reusing the variables
    with tf.variable_scope('DenseNet', reuse=reuse):
        x = tf.layers.dense(x_in, NUM_INPUT)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)

        x = tf.layers.dense(x, 2048)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)

        x = tf.layers.dense(x, 1024)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        # Output layer, class prediction
        out = tf.layers.dense(x, n_classes)
        # Because 'softmax_cross_entropy_with_logits' loss already apply
        # softmax, we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out
    return out


def main():
    """Trains network"""
    # Place all ops on gpu 0 by default
    catar = Catar(path_run=RUN_PATH, path_labels=LABEL_DIR,
                  path_extracted=os.path.join(OUTDIR, 'extracted'),
                  cls_lst=CLS_LST)
    with tf.device('/gpu:0'):
        tower_grads = []
        reuse_vars = False
        # tf Graph input
        X = tf.placeholder(tf.float32, [None, NUM_INPUT])
        Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        # Loop over all GPUs and construct their own computation graph
        for i in range(1, NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                # Split data between GPUs
                _x = X[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                _y = Y[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.
                # Create a graph for training
                logits_train = build_net(_x, NUM_CLASSES, DROPOUT,
                                         reuse=reuse_vars, is_training=True)
                # Create another graph for testing that reuse the same weights
                logits_test = build_net(_x, NUM_CLASSES, DROPOUT,
                                        reuse=True, is_training=False)
    # Define loss and optimizer (with train logits, for dropout to take effect)
                loss_op = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits_train, labels=_y))
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                grads = optimizer.compute_gradients(loss_op)
                # Only second GPU compute accuracy
                if i == 1:
                    # Evaluate model (with test logits, disable dt)
                    correct_pred = tf.equal(
                        tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred, tf.float32))
                reuse_vars = True
                tower_grads.append(grads)
        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)
        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            for step in range(1, NUM_STEPS + 1):
                # Get a batch for each GPU
                batch_x, batch_y = catar.train_next_batch(
                    BATCH_SIZE * NUM_GPUS)
                # Run optimization op (backprop)
                ts = time.time()
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                te = time.time() - ts
                if step % DISPALY_STEP == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy],
                                         feed_dict={X: batch_x, Y: batch_y})
                    out_prnt = ("Step {} : ".format(step)
                                + "Minibatch Loss= {:.4f}, ".format(loss)
                                + "Training Accuracy= {:.3f}, ".format(acc)
                                + ", {} Examples/sec".format(
                                    int(len(batch_x) / te)))
                    print(out_prnt)
                step += 1
            print("Optimization Finished!")
            # Calculate accuracy for 1000 mnist test images
            print("Testing Accuracy:",
                  np.mean(
                      [sess.run(accuracy,
                                feed_dict={
                                    X: catar.test.frames[
                                        i:i + BATCH_SIZE],
                                    Y: catar.test.labels[
                                        i:i + BATCH_SIZE]})
                       for i in range(0, len(catar.test.images), BATCH_SIZE)]))
            return exitfunc()


def average_gradients(tower_grads):
    """function to average gradients from different gpus"""
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Catar(object):
    """Modeled after:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    """

    def __init__(self, path_run=RUN_PATH, path_labels=LABEL_DIR,
                 path_extracted=os.path.join(OUTDIR, 'extracted'),
                 cls_lst=CLS_LST):
        self.train.frames = load_stuff(['frame'],
                                       path_run=os.path.join(
                                           path_run, 'learn'))
        self.train.labels = load_stuff(['label'],
                                       path_run=os.path.join(
                                           path_run, 'learn'))
        self.test.frames = load_stuff(['frame'],
                                      path_run=os.path.join(path_run, 'vldt'))
        self.test.label = load_stuff(['label'],
                                     path_run=os.path.join(path_run, 'vldt'))
        self.test_ind = 0
        self.train_ind = 0

    def _next_batch(self, request_size, arr, ind):
        end = len(arr)
        if (ind + request_size) < end:
            return ind + request_size, arr[ind:ind + request_size]
        else:
            rtns = (range(end - request_size, request_size - end, 1),)
            others = tuple([range(arr.shape[i]) for i in arr.shape[1:]])
            return end - request_size, arr[np.ix_(rtns + others)]

    def train_next_batch(self, request_size):
        _, frame_arr = self._next_batch(request_size, self.train.frames,
                                        self.train_ind)
        self.train_ind, label_arr = self._next_batch(request_size,
                                                     self.train.labels,
                                                     self.train_ind)
        return frame_arr, label_arr

    def test_next_batch(self, request_size):
        _, frame_arr = self._next_batch(request_size, self.test.frames,
                                        self.test_ind)
        self.test_ind, label_arr = self._next_batch(request_size,
                                                    self.test.labels,
                                                    self.test_ind)
        return frame_arr, label_arr


def load_stuff(cmd=['label', 'frame'],
               path_run=RUN_PATH, path_labels=LABEL_DIR,
               path_extracted=os.path.join(OUTDIR, 'extracted'),
               cls_lst=CLS_LST):
    label_arr = []
    frame_arr = []
    runs = pickle.load(open(path_run, 'rb'))
    srcs = np.array([elm[0] for elm in runs])
    for src in np.unique(srcs):
        runs_here = [mem[1] for mem in
                     filter(lambda elm: True if elm[0] == src else False,
                            runs)]
        if 'label' in cmd:
            file_df = pd.read_csv(os.path.join(
                path_labels, src + '.csv'), index_col=0)
        if 'frame' in cmd:
            src_arr = np.load(os.path.join(path_extracted, src + '.npy'))
        for run in runs_here:
            if 'label' in cmd:
                label_arr.append(
                    file_df.loc[run[0]:run[1], cls_lst].values)
            if 'frame' in cmd:
                frame_arr.append(src_arr[run[0]:run[1]])
    if ('label' in cmd) and (len(cmd) == 1):
        return np.hstack(label_arr)
    if ('frame' in cmd) and (len(cmd) == 1):
        return np.vstack(frame_arr)
    return np.vstack(frame_arr), np.vstack(label_arr)


def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)
