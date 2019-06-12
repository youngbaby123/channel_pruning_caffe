#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'

import argparse

from tools.config import cfg, cfg_from_file, cfg_from_list
from tools.channel_pruning_util_ import Prune_parameters_, Net_composition_, Get_pruning_step_group, Prune_tools, \
    get_BN_index_scale, get_BN_index, get_all_keep_index, Get_pruning_singlelayer_step, pruned_channel_forward_by_scale
from tools.train import train_net
from tools.net_util_ import load_net_scale
import copy
import logging


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Classfication network')
    parser.add_argument('--test_prototxt', dest='model_file',
                        help='test prototxt without ext.  i.e. test_caffe_proto',
                        default=None, type=str)
    parser.add_argument('--train_prototxt', dest='model_file_train',
                        help='train prototxt without ext.  i.e. train_caffe_proto',
                        default=None, type=str)
    parser.add_argument('--solver_prototxt', dest='solver_file',
                        help='train prototxt without ext.  i.e. train_caffe_proto',
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
    parser.add_argument('--task_root', dest='task_root',
                        help='the root dir of model and weights files',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args




def demo():
    args = parse_args()
    print(args)

    args.cfg_file = "./test_yxx_resize.yml"
    args.gpu_id = 3
    args.img_list_file = "./test_pad_30_resize_112.txt"
    args.img_root_path = "/home_1/data/caffe/DTY_Side"

    args.task_root = "./model_scale_test"
    args.model_file = "test_res18"
    args.model_file_train = "train_res18"
    args.weights_file = "weights_resize_res18"
    args.solver_file = "solver_res18"

    cfg_from_file(args.cfg_file)
    start_iter = 0

    # model_file = "./model_scale_test/test_res18.prototxt"
    # weights_file = "./model_scale_test/weights_resize_res18.caffemodel"
    # batch_size = 512
    # pruning_num = 5

    # net = load_net_scale(model_file, weights_file=weights_file, GPU_index=3, forward_type="test")


    P_parameters = Prune_parameters_(args)

    Net_composition = Net_composition_(P_parameters.model_file)
    weights_layers_graph = Net_composition.weights_layers_graph

    # weights_layer = weights_layers_graph.keys()[0]
    #
    # pruned_channel_forward_by_scale(net, weights_layers_graph, weights_layer, pruning_num)

    #ignore_layers_ = ['conv1']
    ignore_layers_ = None
    pruning_step_group = Get_pruning_step_group(P_parameters.model_file, ignore_layers=ignore_layers_, order=-1)
    for i in range(48):
        for index_, pruning_dict_ in enumerate(pruning_step_group):

            iter_num = i*len(pruning_step_group) + index_
            # print iter_num
            # logging.error(P_parameters.model_file)
            # logging.error(P_parameters.model_file_train)
            # logging.error(P_parameters.weights_file)
            # logging.error(P_parameters.solver_file)
            # logging.error("========================")
            # logging.error(P_parameters.new_model_file)
            # logging.error(P_parameters.new_model_file_train)
            # logging.error(P_parameters.new_weights_file)
            # logging.error(P_parameters.new_solver_file)
            if iter_num < start_iter + 1:
                P_parameters.update_file(iter_num + 1)
                continue



            layers_list = [layer_i for layer_i in pruning_dict_.keys()]
            pruning_dict = Get_pruning_singlelayer_step(P_parameters.model_file, layers_list)

            # 新一轮模型初始化
            Prune_ = Prune_tools(P_parameters)
            # Prune_.set_device()
            Prune_.update_model_file(pruning_dict)
            Prune_.create_new_net()

            # 使用BN层参数选取剪枝channel
            # BN_keep_index_dict = get_BN_index(Prune_.ori_net, weights_layers_graph, pruning_dict)
            BN_keep_index_dict = get_BN_index_scale(Prune_.ori_net, weights_layers_graph, pruning_dict)
            # print "!!!!!!!!!!!!!!!!!!!!", BN_keep_index_dict
            Weights_pruning_dict = get_all_keep_index(BN_keep_index_dict, pruning_dict, weights_layers_graph)

            # 根据需要剪枝的channel组成的 dict对输入网络进行剪枝
            Prune_.update_new_net(Weights_pruning_dict)
            Prune_.save_new_weights()

            # get fine-tune new model
            solver_prototxt = "/home_1/code/caffe_test/compress/weights_pruning/20190603/model_scale_test/solver_res18_{}.prototxt".format(iter_num)
            output_dir = "/home_1/code/caffe_test/compress/weights_pruning/20190603/model_scale_test"
            pretrained_ = "/home_1/code/caffe_test/compress/weights_pruning/20190603/model_scale_test/weights_resize_res18_{}.caffemodel".format(iter_num)
            last_model_path, if_pruning = train_net(solver_prototxt, output_dir,
                     pretrained_model=pretrained_, max_iters=6000)
            # check if fine-tune new model ok
            print("TODO if use fine-tune model")
            # Prune_.set_device()
            Prune_.save_pruning(last_model_path, IF_PRUNING=if_pruning)

            P_parameters.update_file(iter_num + 1)


    # print (pruning_step_group)

"""
程序入口
"""
if __name__ == '__main__':
    demo()
