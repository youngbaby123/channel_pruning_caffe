#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'

import _init_paths
import argparse
import numpy as np
import cv2
import os
import time
from config import cfg, cfg_from_file, cfg_from_list
from net_util_ import load_net_, change_net_batchsize
from channel_pruning_util_ import Net_composition_, get_all_keep_index, Prune_tools, Prune_parameters_
from datafactory import load_data_batch_with_label, load_data_batch

from collections import OrderedDict
import copy

import pickle
import random

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


def single_batch_forward(net, single_input_data, start = None, end = None):
    net.blobs['data'].data[...] = np.array(single_input_data)
    # time_0 = time.time()
    if start != None and end != None:
        output = net.forward(start=start, end=end)
    elif start != None:
        output = net.forward(start=start)
    elif end != None:
        output = net.forward(end=end)
    else:
        output = net.forward()
    # time_1 = time.time()
    # print ("time: {}".format(time_1 - time_0))
    return output, net


def get_feature_information(net, model_file, input_data):
    net_com = Net_composition_(model_file)
    layer_relation = net_com.get_layer_relation()
    feature_v = OrderedDict()

    for index_, layer_i in enumerate(layer_relation.keys()):
        feature_v[layer_i] = {}
        if index_ == 0:
            feature_v[layer_i]["feature_in"] = {"data": net.blobs["data"].data[...]}
            out_put, net = single_batch_forward(net, input_data, end=layer_relation[layer_i]["name"])
        else:
            feature_v[layer_i]["feature_in"] = OrderedDict()
            for bottom_i in layer_relation[layer_i]["connect_bottom"]:
                #  这里有坑！  多个输出时会有问题的
                feature_v[layer_i]["feature_in"][bottom_i]= feature_v[bottom_i]["feature_out"]
            out_put, net = single_batch_forward(net, input_data, end=layer_relation[layer_i]["name"])

        # ou = out_put[out_put.keys()[-1]]
        # n,c,h,w = ou.shape
        # print "=================================="
        # print layer_i
        # print ou[0, c/2, h/2, :]


        feature_v[layer_i]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
        # print "=================================="
        # print layer_i
        # print feature_v[layer_i]["feature_out"][0, 10, 10, :]
        feature_v[layer_i]["type"] = layer_relation[layer_i]["type"]
        feature_v[layer_i]["weights"] = []
        if layer_i in net.params.keys():
            for layer_params_i in net.params[layer_i]:
                feature_v[layer_i]["weights"].append(layer_params_i.data)
    return feature_v

'''
feature_v["input"]
feature_v["whole_prob"]

feature_v[weights_layer_i]:

feature_v[weights_layer_i]["output_num"]
feature_v[weights_layer_i]["type"]
feature_v[weights_layer_i]["Weights"]
feature_v[weights_layer_i]["BN"]
feature_v[weights_layer_i]["Scale"]
feature_v[weights_layer_i]["ReLU"]
feature_v[weights_layer_i]["channel_prob"]

feature_v[weights_layer_i]["Weights"]["feature_out"]
feature_v[weights_layer_i]["Weights"]["parameter"]
feature_v[weights_layer_i]["BN"]["feature_out"]
feature_v[weights_layer_i]["BN"]["parameter"]
feature_v[weights_layer_i]["Scale"]["feature_out"]
feature_v[weights_layer_i]["Scale"]["parameter"]
feature_v[weights_layer_i]["ReLU"]["feature_out"]
'''


def get_pruning_prob_each(net, model_file, input_data):
    net_com = Net_composition_(model_file)
    weights_layers_graph = net_com.weights_layers_graph
    # layer_relation = net_com.get_layer_relation()

    feature_v = OrderedDict()
    feature_v["input"] = {"data": net.blobs["data"].data[...]}
    out_put, net = single_batch_forward(net, input_data)
    feature_v["whole_prob"] = copy.deepcopy(out_put[out_put.keys()[-1]])
    # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

    for index_, weights_layer_i in enumerate(weights_layers_graph.keys()):
        print("Start test the weights layer: {}\tID: {}".format(weights_layer_i, index_))
        feature_v[weights_layer_i] = {}
        feature_v[weights_layer_i]["output_num"] = weights_layers_graph[weights_layer_i]["output_num"]
        feature_v[weights_layer_i]["type"] = weights_layers_graph[weights_layer_i]["type"]
        feature_v[weights_layer_i]["Weights"] = {}
        feature_v[weights_layer_i]["BN"] = {}
        feature_v[weights_layer_i]["Scale"] = {}
        feature_v[weights_layer_i]["ReLU"] = {}

        feature_v[weights_layer_i]["channel_prob"] = []

        # weights blobs new_net.params[fc_name][1].data[...]
        out_put, net = single_batch_forward(net, input_data, end=weights_layer_i)
        feature_v[weights_layer_i]["Weights"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
        feature_v[weights_layer_i]["Weights"]["parameter"] = []
        for i in range(len(net.params[weights_layer_i])):
            feature_v[weights_layer_i]["Weights"]["parameter"].append(net.params[weights_layer_i][i].data[...])

        # BN
        if (weights_layers_graph[weights_layer_i]["BN"] != []):
            BN_name = weights_layers_graph[weights_layer_i]["BN"]
            out_put, net = single_batch_forward(net, input_data, end=BN_name)
            feature_v[weights_layer_i]["BN"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["BN"]["parameter"] = []
            for i in range(len(net.params[BN_name])):
                feature_v[weights_layer_i]["BN"]["parameter"].append(net.params[BN_name][i].data[...])

        # Scale
        if (weights_layers_graph[weights_layer_i]["Scale"] != []):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            out_put, net = single_batch_forward(net, input_data, end=Scale_name)
            feature_v[weights_layer_i]["Scale"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["Scale"]["parameter"] = []
            for i in range(len(net.params[Scale_name])):
                feature_v[weights_layer_i]["Scale"]["parameter"].append(net.params[Scale_name][i].data[...])

        # ReLU
        if (weights_layers_graph[weights_layer_i]["ReLU"] != []):
            ReLU_name = weights_layers_graph[weights_layer_i]["ReLU"]
            out_put, net = single_batch_forward(net, input_data, end=ReLU_name)
            feature_v[weights_layer_i]["ReLU"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])

        if (weights_layers_graph[weights_layer_i]["Scale"] != []):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            scale_tmp_0 = 0
            scale_tmp_1 = 0
            for i in range(feature_v[weights_layer_i]["output_num"]):
                # print("Test channel {}.".format(i))
                if i != 0:
                    net.params[Scale_name][0].data[i - 1] = scale_tmp_0
                    net.params[Scale_name][1].data[i - 1] = scale_tmp_1
                # print("============================================")
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))
                scale_tmp_0 = net.params[Scale_name][0].data[i]
                scale_tmp_1 = net.params[Scale_name][1].data[i]
                net.params[Scale_name][0].data[i] = 0
                net.params[Scale_name][1].data[i] = 0
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))

                out_put, net = single_batch_forward(net, input_data)
                feature_v[weights_layer_i]["channel_prob"].append(copy.deepcopy(out_put[out_put.keys()[-1]]))
                # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

            net.params[Scale_name][0].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_0
            net.params[Scale_name][1].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_1

    return feature_v

def get_pruning_prob_each_1(net, model_file, input_data):
    net_com = Net_composition_(model_file)
    weights_layers_graph = net_com.weights_layers_graph
    # layer_relation = net_com.get_layer_relation()

    feature_v = OrderedDict()
    feature_v["input"] = {"data": net.blobs["data"].data[...]}
    out_put, net = single_batch_forward(net, input_data)
    feature_v["whole_prob"] = copy.deepcopy(out_put[out_put.keys()[-1]])
    # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

    for index_, weights_layer_i in enumerate(weights_layers_graph.keys()):
        print("Start test the weights layer: {}\tID: {}".format(weights_layer_i, index_))
        feature_v[weights_layer_i] = {}
        feature_v[weights_layer_i]["output_num"] = weights_layers_graph[weights_layer_i]["output_num"]
        feature_v[weights_layer_i]["type"] = weights_layers_graph[weights_layer_i]["type"]
        feature_v[weights_layer_i]["Weights"] = {}
        feature_v[weights_layer_i]["BN"] = {}
        feature_v[weights_layer_i]["Scale"] = {}
        feature_v[weights_layer_i]["ReLU"] = {}

        feature_v[weights_layer_i]["channel_prob"] = []

        # weights blobs new_net.params[fc_name][1].data[...]
        out_put, net = single_batch_forward(net, input_data, end=weights_layer_i)
        feature_v[weights_layer_i]["Weights"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
        feature_v[weights_layer_i]["Weights"]["parameter"] = []
        for i in range(len(net.params[weights_layer_i])):
            feature_v[weights_layer_i]["Weights"]["parameter"].append(net.params[weights_layer_i][i].data[...])

        # BN
        if(weights_layers_graph[weights_layer_i]["BN"] !=[]):
            BN_name = weights_layers_graph[weights_layer_i]["BN"]
            out_put, net = single_batch_forward(net, input_data, end=BN_name)
            feature_v[weights_layer_i]["BN"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["BN"]["parameter"] = []
            for i in range(len(net.params[BN_name])):
                feature_v[weights_layer_i]["BN"]["parameter"].append(net.params[BN_name][i].data[...])

        # Scale
        if(weights_layers_graph[weights_layer_i]["Scale"] !=[]):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            out_put, net = single_batch_forward(net, input_data, end=Scale_name)
            feature_v[weights_layer_i]["Scale"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["Scale"]["parameter"] = []
            for i in range(len(net.params[Scale_name])):
                feature_v[weights_layer_i]["Scale"]["parameter"].append(net.params[Scale_name][i].data[...])

        # ReLU
        if(weights_layers_graph[weights_layer_i]["ReLU"] !=[]):
            ReLU_name = weights_layers_graph[weights_layer_i]["ReLU"]
            out_put, net = single_batch_forward(net, input_data, end=ReLU_name)
            feature_v[weights_layer_i]["ReLU"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])

        if (weights_layers_graph[weights_layer_i]["Scale"] != []):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            scale_tmp_0 = 0
            scale_tmp_1 = 0
            for i in range(feature_v[weights_layer_i]["output_num"]):
                print("Test channel {}.".format(i))
                if i != 0:
                    net.params[Scale_name][0].data[i - 1] = scale_tmp_0
                    net.params[Scale_name][1].data[i - 1] = scale_tmp_1
                # print("============================================")
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))
                scale_tmp_0 = net.params[Scale_name][0].data[i]
                scale_tmp_1 = net.params[Scale_name][1].data[i]
                net.params[Scale_name][0].data[i] = 0
                net.params[Scale_name][1].data[i] = 0
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))

                out_put, net = single_batch_forward(net, input_data)
                feature_v[weights_layer_i]["channel_prob"].append(copy.deepcopy(out_put[out_put.keys()[-1]]))
                # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

            net.params[Scale_name][0].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_0
            net.params[Scale_name][1].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_1

    return feature_v

def get_pruning_prob_each_2(net, model_file, input_data):
    net_com = Net_composition_(model_file)
    weights_layers_graph = net_com.weights_layers_graph
    # layer_relation = net_com.get_layer_relation()

    feature_v = OrderedDict()
    feature_v["input"] = {"data": net.blobs["data"].data[...]}
    out_put, net = single_batch_forward(net, input_data)
    feature_v["whole_prob"] = copy.deepcopy(out_put[out_put.keys()[-1]])
    # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

    for index_, weights_layer_i in enumerate(weights_layers_graph.keys()):
        print("Start test the weights layer: {}\tID: {}".format(weights_layer_i, index_))
        feature_v[weights_layer_i] = {}
        feature_v[weights_layer_i]["output_num"] = weights_layers_graph[weights_layer_i]["output_num"]
        feature_v[weights_layer_i]["type"] = weights_layers_graph[weights_layer_i]["type"]
        feature_v[weights_layer_i]["Weights"] = {}
        feature_v[weights_layer_i]["BN"] = {}
        feature_v[weights_layer_i]["Scale"] = {}
        feature_v[weights_layer_i]["ReLU"] = {}

        feature_v[weights_layer_i]["channel_prob"] = []

        # weights blobs new_net.params[fc_name][1].data[...]
        out_put, net = single_batch_forward(net, input_data, end=weights_layer_i)
        feature_v[weights_layer_i]["Weights"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
        feature_v[weights_layer_i]["Weights"]["parameter"] = []
        for i in range(len(net.params[weights_layer_i])):
            feature_v[weights_layer_i]["Weights"]["parameter"].append(net.params[weights_layer_i][i].data[...])

        # BN
        if (weights_layers_graph[weights_layer_i]["BN"] != []):
            BN_name = weights_layers_graph[weights_layer_i]["BN"]
            out_put, net = single_batch_forward(net, input_data, end=BN_name)
            feature_v[weights_layer_i]["BN"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["BN"]["parameter"] = []
            for i in range(len(net.params[BN_name])):
                feature_v[weights_layer_i]["BN"]["parameter"].append(net.params[BN_name][i].data[...])

        # Scale
        if (weights_layers_graph[weights_layer_i]["Scale"] != []):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            out_put, net = single_batch_forward(net, input_data, end=Scale_name)
            feature_v[weights_layer_i]["Scale"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])
            feature_v[weights_layer_i]["Scale"]["parameter"] = []
            for i in range(len(net.params[Scale_name])):
                feature_v[weights_layer_i]["Scale"]["parameter"].append(net.params[Scale_name][i].data[...])

        # ReLU
        if (weights_layers_graph[weights_layer_i]["ReLU"] != []):
            ReLU_name = weights_layers_graph[weights_layer_i]["ReLU"]
            out_put, net = single_batch_forward(net, input_data, end=ReLU_name)
            feature_v[weights_layer_i]["ReLU"]["feature_out"] = copy.deepcopy(out_put[out_put.keys()[-1]])

        if (weights_layers_graph[weights_layer_i]["Scale"] != []):
            Scale_name = weights_layers_graph[weights_layer_i]["Scale"]
            scale_tmp_0 = 0
            scale_tmp_1 = 0
            for i in range(feature_v[weights_layer_i]["output_num"]):
                print("Test channel {}.".format(i))
                if i != 0:
                    net.params[Scale_name][0].data[i - 1] = scale_tmp_0
                    net.params[Scale_name][1].data[i - 1] = scale_tmp_1
                # print("============================================")
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))
                scale_tmp_0 = net.params[Scale_name][0].data[i]
                scale_tmp_1 = net.params[Scale_name][1].data[i]
                net.params[Scale_name][0].data[i] = 0
                net.params[Scale_name][1].data[i] = 0
                # print("scale: {}.".format(net.params[Scale_name][0].data[...]))

                out_put, net = single_batch_forward(net, input_data)
                feature_v[weights_layer_i]["channel_prob"].append(copy.deepcopy(out_put[out_put.keys()[-1]]))
                # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))

            net.params[Scale_name][0].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_0
            net.params[Scale_name][1].data[feature_v[weights_layer_i]["output_num"] - 1] = scale_tmp_1

    return feature_v

# 对比只将scale的值变成0以及weights砍掉  结果是否一样
def my_test():
    args = parse_args()
    print(args)

    args.cfg_file = "../test_yxx_resize.yml"
    args.gpu_id = 3
    args.img_list_file = "../test_pad_30_resize_112.txt"
    args.img_root_path = "/home_1/data/caffe/DTY_Side"

    args.task_root = "../model_2"
    args.model_file = "test_res18"
    args.model_file_train = "train_res18"
    args.weights_file = "weights_resize_res18"
    args.solver_file = "solver_res18"

    batch_size = 128

    cfg_from_file(args.cfg_file)
    P_parameters = Prune_parameters_(args)

    net_com = Net_composition_(P_parameters.model_file)
    weights_graph = net_com.weights_layers_graph

    pruning_layer_dict = {"conv1": 1}
    BN_pruning_dict = {"bn_conv1": [i for i in range(63)]}

    Prune_ = Prune_tools(P_parameters)
    Prune_.update_model_file(pruning_layer_dict)
    Prune_.create_new_net()


    Weights_pruning_dict = get_all_keep_index(BN_pruning_dict, pruning_layer_dict, weights_graph)

    Prune_.update_new_net(Weights_pruning_dict)
    Prune_.save_new_weights()





    model_ori = "/home_1/code/caffe_test/compress/weights_pruning/20190529/model_2/test_res18.prototxt"
    weights_ori = "/home_1/code/caffe_test/compress/weights_pruning/20190529/model_2/weights_resize_res18.caffemodel"
    net_scale = load_net_(model_ori, weights_file=weights_ori, GPU_index=3, batch_size=batch_size, forward_type="test")
    net_scale.params["scale_conv1"][0].data[-1] = 0
    net_scale.params["scale_conv1"][1].data[-1] = 0

    net_ori = load_net_(model_ori, weights_file=weights_ori, GPU_index=3, batch_size=batch_size, forward_type="test")

    model_pruning = "/home_1/code/caffe_test/compress/weights_pruning/20190529/model_2/test_res18_0.prototxt"
    weights_pruning = "/home_1/code/caffe_test/compress/weights_pruning/20190529/model_2/weights_resize_res18_0.caffemodel"
    net_pruning = load_net_(model_pruning, weights_file=weights_pruning, GPU_index=3, batch_size=batch_size, forward_type="test")

    img_file_list = open(args.img_list_file, "r").readlines()
    img_list = img_file_list[0:batch_size]
    input_data, input_label, input_data_name = load_data_batch_with_label(net_pruning, args.img_root_path, img_list)


    out_put_pruning, net_pruning = single_batch_forward(net_pruning, input_data)
    out_put_scale, net_scale = single_batch_forward(net_scale, input_data)
    out_put_ori, net_ori = single_batch_forward(net_ori, input_data)
    # print "======================="
    # print "pruning: "
    # print out_put_pruning
    # print "-----------------------"
    # print "scale: "
    # print out_put_scale
    # print "-----------------------"
    for key_ in out_put_pruning.keys():
        for i in range(len(out_put_pruning[key_])):
            print "------------------------"
            print "scale: "
            print out_put_scale[key_][i]
            print "pruning: "
            print out_put_pruning[key_][i]
            print "ori: "
            print out_put_ori[key_][i]





def demo_ori():
    model_file = "../test_res18.prototxt"
    weights_file = "../weights_resize_res18.caffemodel"
    data_file = "../test_pad_30_resize_112.txt"
    img_root = "/home_1/data/caffe/DTY_Side"
    cfg_file = "../test_yxx_resize.yml"
    pkl_dir = "../pkl_feature"
    batch_size = 512

    cfg_from_file(cfg_file)

    net = load_net_(model_file, weights_file=weights_file, GPU_index=3, batch_size=batch_size, forward_type="test")

    img_file_list = open(data_file, "r").readlines()

    img_list = img_file_list[0:batch_size]

    input_data, input_label, input_data_name = load_data_batch_with_label(net, img_root, img_list)
    feature_v = get_feature_information(net, model_file, input_data)
    # print feature_v["prob"]

    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    save_name = os.path.join(pkl_dir, "feature_res_{}.pkl".format(batch_size))
    pickle.dump(feature_v, open(save_name, 'wb'))

    # # for i in net.blobs:
    # #     print i
    # #
    # # print "============================"
    # # for i in net.params:
    # #     print i
    #
    # # for layer_name, blob in net.blobs.iteritems():
    # #     print layer_name
    # # #     print blob.data[...]
    # net_com = Net_composition_(model_file)
    # layer_relation = net_com.get_layer_relation()
    # # for i in layer_relation.keys()[:5]:
    # #     print "--------------------------------------------------"
    # #     print i
    # #     print layer_relation[i]
    # # params_list[]
    # # for layer_name, params in net.params.iteritems():
    # #     print layer_name
    # #     # for params_i in params:
    # #     #     print params_i.data
    # #
    #
    # feature_v = OrderedDict()
    # img_file_list = open(data_file, "r").readlines()
    # # for file_i in img_file_list[0:1]:
    # #     img_list = [file_i]
    # #     input_data, input_label, input_data_name = load_data_batch_with_label(net, img_root, img_list)
    # #
    # #     out_put = net.blobs["data"].data[...]
    # #     for index_, layer_i in enumerate(layer_relation.keys()):
    # #         print "--------------------------------------------------"
    # #         print layer_i
    # #         feature_v[layer_i] = {}
    # #         feature_v[layer_i]["feature_in"] = out_put
    # #         out_put, net = single_forward(net, input_data, start=layer_relation[layer_i]["top"],
    # #                                       end=layer_relation[layer_i]["name"])
    # #         feature_v[layer_i]["feature_out"] = out_put
    # #         feature_v[layer_i]["type"] = layer_relation[layer_i]["type"]
    # #         feature_v[layer_i]["weights"] = []
    # #         if layer_i in net.params.keys():
    # #             for layer_params_i in net.params[layer_i]:
    # #                 feature_v[layer_i]["weights"].append(layer_params_i.data)
    # #     print feature_v


if __name__ == '__main__':
    model_file = "../test_res18.prototxt"
    weights_file = "../weights_resize_res18.caffemodel"
    data_file = "../test_pad_30_resize_112.txt"
    img_root = "/home_1/data/caffe/DTY_Side"
    cfg_file = "../test_yxx_resize.yml"
    pkl_dir = "../pkl_feature"
    batch_size = 512

    cfg_from_file(cfg_file)

    net = load_net_(model_file, weights_file=weights_file, GPU_index=3, batch_size=batch_size, forward_type="test")

    img_file_list = open(data_file, "r").readlines()

    random.shuffle(img_file_list)
    img_list = img_file_list[0:batch_size]

    input_data, input_label, input_data_name = load_data_batch_with_label(net, img_root, img_list)
    print input_data_name
    print input_label
    feature_v = get_pruning_prob_each(net, model_file, input_data)

    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    save_name = os.path.join(pkl_dir, "feature_res_channel_{}.pkl".format(batch_size))
    pickle.dump(feature_v, open(save_name, 'wb'))

    # my_test()

