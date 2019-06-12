#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'

import logging
import caffe
import numpy as np
import os

from config import cfg, cfg_from_file, cfg_from_list
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from google.protobuf import text_format

import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        caffe.set_device(2)
        caffe.set_mode_gpu()
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            # self.solver.net = pretrained_model

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)


    def snapshot(self):

        net = self.solver.net

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        loss_list = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            single_step_loss = self.solver.net.blobs["loss"].data
            if self.solver.iter < cfg.TRAIN.CALLBACK_ITERS:
                loss_list.append(single_step_loss)
            else:
                del(loss_list[0])
                loss_list.append(single_step_loss)
                mean_loss = np.mean(loss_list)
                if mean_loss < cfg.TRAIN.CALLBACK_VALUE * cfg.TRAIN.CALLBACK_STEP_SCALE:
                    break

            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if_pruning = True
        mean_loss_final = np.mean(loss_list)
        if mean_loss_final > cfg.TRAIN.CALLBACK_VALUE*cfg.TRAIN.CALLBACK_FINAL_SCALE:
            if_pruning = False
        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        #return model_paths
        return self.snapshot() ,if_pruning


def train_net(solver_prototxt,  output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Classfication network."""

    sw = SolverWrapper(solver_prototxt, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    #model_paths = sw.train_model(max_iters)
    last_model_path, if_pruning = sw.train_model(max_iters)
    print 'done solving'
    #return model_paths
    return last_model_path, if_pruning
