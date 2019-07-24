#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import os

import numpy as np
import scipy.io as sio
import tensorflow as tf

whereAmI = os.path.realpath(__file__)
whereAmI_folder = os.path.dirname(whereAmI)

slc = sio.loadmat(whereAmI_folder + '/flint-data-preprocessing/flint_procd.mat')

nRuns = 6

rmse = np.zeros([nRuns])
maae = np.zeros([nRuns])

for iRun in range(nRuns):

    x = slc['procd'][0][iRun]['spikes']
    z = slc['procd'][0][iRun]['velocities']

    x0 = x[:5000, ]
    z0 = z[:5000, ]
    x1 = x[5000:6000, ]
    z1 = z[5000:6000, ]

    dx = x0.shape[1]
    dz = z0.shape[1]

    n_steps = 3
    n_neurons = 20
    batch_size = 1000


    def obs_hist(ind):
        ind = np.array(ind).flatten()
        neur = np.zeros((ind.size, n_steps, dx))
        for i0 in range(ind.size):
            s_idx = range(ind[i0] - n_steps + 1, ind[i0] + 1)
            neur[i0, :, :] = x[s_idx, :]
        return neur


    def hid_hist(ind):
        return z[np.array(ind), :]


    g = tf.Graph()  # this graph is for building features

    # for developing, try: sess = tf.InteractiveSession()
    # Tell TensorFlow that the model will be built into the default Graph.
    with g.as_default():
        tf.set_random_seed(42)  # for repeatability

        with tf.name_scope('keep_prob'):
            keep_prob_in_ = tf.placeholder("float")
            keep_prob_out_ = tf.placeholder("float")

        # Generate placeholders for the images and labels.
        with tf.name_scope('inputs'):
            neural_ = tf.placeholder(tf.float32, shape=[batch_size, n_steps, dx])
            neural_dropped = tf.nn.dropout(neural_, keep_prob=keep_prob_in_)
            neural_split = [tf.reshape(v, shape=[-1, dx]) for v in tf.split(neural_, n_steps, 1)]

        with tf.name_scope('targets'):
            velocities_ = tf.placeholder(tf.float32, shape=[batch_size, dz])


        def lstm_step(inp, prev, state):
            with tf.name_scope('dimensionality'):
                dstate = state.get_shape()[1].__int__()
                din = inp.get_shape()[1].__int__()
                dout = prev.get_shape()[1].__int__()
                gates = {}
            for g in ['forget', 'input', 'output', 'state']:
                with tf.name_scope(g):
                    W = tf.Variable(tf.truncated_normal([din, dstate], stddev=1 / tf.sqrt(tf.to_float(din))))
                    U = tf.Variable(tf.truncated_normal([dout, dstate], stddev=1 / tf.sqrt(tf.to_float(dout))))
                    b = tf.Variable(tf.zeros([1, dstate]))
                    combo = tf.matmul(inp, W) + tf.matmul(prev, U) + b
                    if g in ['forget', 'input', 'output']:
                        gates[g] = tf.sigmoid(combo)
                    else:
                        state = tf.multiply(gates['forget'], state) + tf.multiply(gates['input'], tf.tanh(combo))
            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([dstate, dout], stddev=1 / tf.sqrt(tf.to_float(dstate))))
                b = tf.Variable(tf.zeros([1, dout]))
                outp = tf.matmul(tf.multiply(gates['output'], tf.tanh(state)), W) + b
            return outp, state


        def lstm_full(inp, dout, dstate):
            with tf.name_scope('dimensionality'):
                T = len(inp)
                n = inp[0].get_shape()[0]
            with tf.name_scope('states'):
                state = tf.Variable(tf.zeros([n, dstate]), trainable=False)
                out = tf.Variable(tf.zeros([n, dout]), trainable=False)
            for t in range(T):
                with tf.name_scope('step' + str(t)):
                    out, state = lstm_step(inp[t], out, state)
                    if t < T - 1:
                        state = tf.layers.batch_normalization(state)
                        out = tf.layers.batch_normalization(out)
            return out


        with tf.name_scope('lstm'):
            lstm_output = tf.nn.dropout(lstm_full(neural_split, n_neurons, n_neurons), keep_prob=keep_prob_out_)

        with tf.name_scope('outputs'):
            W = tf.Variable(tf.truncated_normal([n_neurons, 2], stddev=1 / np.sqrt(float(n_neurons))))
            b = tf.Variable(tf.zeros([1, 2]))
            output = tf.matmul(lstm_output, W) + b

        with tf.name_scope('loss'):
            mse_loss = tf.reduce_mean(tf.squared_difference(output, velocities_), name='mse')

        optimizer = tf.train.AdadeltaOptimizer(1.)
        train_op = optimizer.minimize(mse_loss)

        with tf.name_scope('validation'):
            val_op = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, velocities_), axis=1))

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a session for training g1
        sess = tf.Session(graph=g)

        # Run the Op to initialize the variables.
        sess.run(init)

        max_train_idx = 5000

        # training
        for reps in range(300):
            for i in range(int(max_train_idx / batch_size)):
                # randomly grab a training set
                idx = np.arange(batch_size * i + n_steps, batch_size * (i + 1) + n_steps)
                # if i % 10 == 0:  # every 10th step we run our validation step to see how we're doing
                #     f_dict = {neural_: obs_hist(idx), velocities_: hid_hist(idx), keep_prob_in_: 1., keep_prob_out_: 1.}
                #     vali = sess.run(val_op, feed_dict=f_dict)
                #     print(vali)
                # do a regular training step
                f_dict = {neural_: obs_hist(idx), velocities_: hid_hist(idx), keep_prob_in_: .5, keep_prob_out_: .95}
                sess.run(train_op, feed_dict=f_dict)

        # testing
        hid_feed = np.zeros((batch_size, dz))
        obs_feed = np.concatenate((obs_hist(np.arange(5000, 5000 + 1000)), np.zeros((batch_size - 1000, n_steps, dx))),
                                  0)

        f_dict = {neural_: obs_feed, velocities_: hid_feed, keep_prob_in_: 1., keep_prob_out_: 1.}

        hid_preds = sess.run(output, feed_dict=f_dict)

        hid_preds = hid_preds[np.arange(1000), :].T
        rmse[iRun] = np.sqrt(np.mean(np.mean(np.square(hid_preds - z1.T))))

        ang_preds = np.arctan2(hid_preds[1,], hid_preds[0,])
        ang_true = np.arctan2(z1.T[1,], z1.T[0,])
        ang_err = ang_preds - ang_true
        maae[iRun] = mean_abs_ang_err = np.mean(
            np.min(np.abs(np.vstack((ang_err - 2 * np.pi, ang_err, ang_err + 2 * np.pi))), axis=0))

        print('run: ' + str(iRun + 1))
        print('RMSE: ' + str(rmse[iRun]))
        print('MAAE: ' + str(maae[iRun]))

with open(os.path.dirname(whereAmI_folder) + 'flint_lstm.txt', 'w') as fi:
    for iRun in range(nRuns):
        fi.write('run: ' + str(iRun + 1) + '\n')
        fi.write('RMSE: ' + str(rmse[iRun]) + '\n')
        fi.write('MAAE: ' + str(maae[iRun]) + '\n')

        # writer = tf.summary.FileWriter('/tmp/', sess.graph)
