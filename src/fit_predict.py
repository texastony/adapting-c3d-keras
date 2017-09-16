from __future__ import print_function
import os
import time
import catar
import numpy as np
import pandas as pd
import tensorflow as tf
from progbar import Progbar

NUM_GPUS = 16
NUM_STEPS = 1
LEARNING_RATE = 0.00003
BATCH_SIZE = 1440
DISPALY_STEP = 10
NUM_INPUT = catar.NUM_INPUT8192
NUM_CLASSES = len(catar.CLS_LST)
DROPOUT = 0.5


def build_net(x_in, n_classes, dropout, reuse, is_training):
    """Creates a dense multi-layer network"""
    act = tf.contrib.keras.activations.relu
    out_act = tf.contrib.keras.activations.sigmoid
    # Define a scope for reusing the variables
    with tf.variable_scope('DenseNet', reuse=reuse):
        x = tf.layers.dense(x_in, 2048, activation=act)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        x = tf.layers.dense(x, 1024, activation=act)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        # Output layer, class prediction
        out = tf.layers.dense(x, n_classes, act=out_act)
    return out


def main(catar):
    """Trains network"""
    # Place all ops on gpu 0 by default
    with tf.device('/cpu:0'):
        tower_grads = []
        reuse_vars = False
        X = tf.placeholder(tf.float32, [None, NUM_INPUT])
        Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                _x = X[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                _y = Y[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                logits_train = build_net(_x, NUM_CLASSES, DROPOUT,
                                         reuse=reuse_vars, is_training=True)
                logits_test = build_net(_x, NUM_CLASSES, DROPOUT,
                                        reuse=True, is_training=False)
                loss_op = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits_train, labels=_y))
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                grads = optimizer.compute_gradients(loss_op)
                if i == 1:
                    correct_pred = tf.equal(
                        tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred, tf.float32))
                reuse_vars = True
                tower_grads.append(grads)
        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)
        init = tf.global_variables_initializer()
        status = Progbar(NUM_STEPS)
        with tf.Session() as sess:
            sess.run(init)
            step = 1
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
                    out_prnt = ("\nMinibatch Loss= {:.4f}, ".format(loss)
                                + "Training Accuracy= {:.3f}, ".format(acc)
                                + ", {} Examples/sec".format(
                                    int(len(batch_x) / te)))
                    step += 1
                status.update(step, text=out_prnt)
            print("Optimization Finished!")
            accur_steps = np.arange(0, len(catar.test_frames), BATCH_SIZE)
            outs = np.empty_like(accur_steps, dtype=np.float)
            status = Progbar(accur_steps[-1])
            for ind, test_ind in enumerate(accur_steps):
                out = sess.run(accuracy, feed_dict={
                    X: catar.test_frames[
                        test_ind:test_ind + BATCH_SIZE],
                    Y: catar.test_labels[
                        test_ind:test_ind + BATCH_SIZE]})
                outs[ind] = out
                status.update(test_ind)
            print("Testing Accuracy:", np.mean(outs))
            return outs  # exitfunc()


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


def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)


os.chdir('..')
catar_here = catar.Catar()
out = main(catar_here)
