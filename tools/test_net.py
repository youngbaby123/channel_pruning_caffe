#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'


import _init_paths
from config import cfg, cfg_from_file, cfg_from_list
from datafactory import load_data_batch_with_label

import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
from net_util_ import load_net_, change_net_batchsize, predict_with_label
import time
import pickle


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Classfication network')
    parser.add_argument('--deploy', dest='deploy_file',
                        help='deploy prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='weights_file',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--img_root', dest='img_root_path',
                        help='img data root path',
                        default=None, type=str)
    parser.add_argument('--img_list', dest='img_list_file',
                        help='img list file',
                        default=None, type=str)
    parser.add_argument('--save_root', dest='save_root_path',
                        help='results pkl save path',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test_net_cls(args):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    net = load_net_(args.deploy_file, args.weights_file, args.gpu_id, batch_size = -1, forward_type = "test")
    img_list = open(args.img_list_file, "r").readlines()

    input_layer_name = net._layer_names[net._inputs[0]]
    batch_size, c, h, w = net.blobs[input_layer_name].shape

    epoch_num = int(len(img_list) / batch_size)

    predict_res = []
    for epoch_i in range(epoch_num):
        if epoch_i % 10 == 0:
            print "TODO: {}% ".format(1.0 * epoch_i / epoch_num * 100)
        batch_img_list = img_list[epoch_i * batch_size:(epoch_i + 1) * batch_size]
        input_data, input_label, input_data_name = load_data_batch_with_label(net, args.img_root_path, batch_img_list)
        single_batch_res = predict_with_label(net, input_data, input_label, input_data_name)
        predict_res.extend(single_batch_res)

    pkl_dir = os.path.join(args.save_root_path, "res_pkl")
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    save_name = os.path.join(pkl_dir, "{}_res.pkl".format(os.path.splitext(os.path.basename(args.weights_file))[0]))
    pickle.dump(predict_res, open(save_name, 'wb'))

if __name__ == '__main__':
    args = parse_args()
    test_net_cls(args)
