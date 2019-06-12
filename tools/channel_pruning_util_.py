#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'

import _init_paths

import numpy as np
import os, sys
import os.path as osp
import google.protobuf as pb
import google.protobuf.text_format
import time
import shutil

from collections import OrderedDict
from net_util_ import load_net_
from config import cfg, cfg_from_file, cfg_from_list
import copy
import logging
import math

import caffe
import argparse


class Net_composition_():
    def __init__(self, model_file, order=1):
        self.model_file = model_file
        self.all_layer_graph = self.get_layer_graph()
        self.weights_layers_graph = self.get_weights_layers_graph()
        self.weights_pruning_group = self.get_weights_pruning_group(order = order)


    def get_layer_graph(self):
        all_layer_graph = OrderedDict()
        with open(self.model_file) as f:
            model_pb = caffe.proto.caffe_pb2.NetParameter()
            pb.text_format.Merge(f.read(), model_pb)

        # 遍历模型中的相应层
        for i, layer in enumerate(model_pb.layer):
            all_layer_graph[layer.name] = layer
        return all_layer_graph

    # 根据layer的type类型返回相同type的layer的 dict
    def get_layer_by_type(self, layer_type):
        layer_tpye_dict = OrderedDict()
        for layer_i in self.all_layer_graph:
            if self.all_layer_graph[layer_i].type == layer_type:
                layer_tpye_dict[layer_i] = self.all_layer_graph[layer_i]
        return layer_tpye_dict

    # 根据layer的type类型返回相同type的layer的 dict
    def get_layer_by_types(self, layer_types):
        layer_tpye_dict = OrderedDict()
        for layer_i in self.all_layer_graph:
            if self.all_layer_graph[layer_i].type in layer_types:
                layer_tpye_dict[layer_i] = self.all_layer_graph[layer_i]
        return layer_tpye_dict

    def get_weights_layer_bottom(self, layer_name, weights_layer_bottom=[]):
        check_layer = self.all_layer_graph[layer_name]
        for bottom_name in check_layer.bottom:
            if self.all_layer_graph[bottom_name].type in ['Convolution', 'InnerProduct']:
                weights_layer_bottom.append(bottom_name)
            else:
                weights_layer_bottom = self.get_weights_layer_bottom(bottom_name, weights_layer_bottom=weights_layer_bottom)
        return weights_layer_bottom

    def get_weights_layer_top(self, layer_name):
        weights_layer_top = []
        conv_layer_dict = self.get_layer_by_type('Convolution')
        for conv_name in conv_layer_dict.keys():
            conv_layer_bottom_tmp = self.get_weights_layer_bottom(conv_name, weights_layer_bottom=[])
            if layer_name in conv_layer_bottom_tmp:
                weights_layer_top.append(conv_name)

        fc_layer_dict = self.get_layer_by_type('InnerProduct')
        for fc_name in fc_layer_dict.keys():
            fc_layer_bottom_tmp = self.get_weights_layer_bottom(fc_name, weights_layer_bottom=[])
            if layer_name in fc_layer_bottom_tmp:
                weights_layer_top.append(fc_name)

        weights_layer_top = list(set(weights_layer_top))
        return weights_layer_top

    def get_BN_for_weights(self, weights_name):
        BN_layer_dict = self.get_layer_by_type('BatchNorm')
        for BN_name in BN_layer_dict.keys():
            if weights_name in BN_layer_dict[BN_name].bottom:
                return BN_name
        return []

    def get_sclae_for_weights(self, weights_name):
        Scale_layer_dict = self.get_layer_by_type('Scale')
        for Scale_name in Scale_layer_dict.keys():
            if weights_name in Scale_layer_dict[Scale_name].bottom:
                return Scale_name
        return []

    def get_relu_for_weights(self, weights_name):
        ReLU_layer_dict = self.get_layer_by_type('ReLU')
        for ReLU_name in ReLU_layer_dict.keys():
            if weights_name in ReLU_layer_dict[ReLU_name].bottom:
                return ReLU_name
        return []

    def get_weights_layers_graph(self):
        Weights_layers_graph = OrderedDict()
        Weights_layer_dict = self.get_layer_by_types(['Convolution', 'InnerProduct'])

        for weights_layer_name in Weights_layer_dict.keys():
            Weights_layers_graph[weights_layer_name] = {}
            Weights_layers_graph[weights_layer_name]["name"] = weights_layer_name
            Weights_layers_graph[weights_layer_name]["BN"] = self.get_BN_for_weights(weights_layer_name)
            Weights_layers_graph[weights_layer_name]["Scale"] = self.get_sclae_for_weights(weights_layer_name)
            Weights_layers_graph[weights_layer_name]["ReLU"] = self.get_relu_for_weights(weights_layer_name)
            Weights_layers_graph[weights_layer_name]["Top_weights"] = self.get_weights_layer_top(weights_layer_name)
            Weights_layers_graph[weights_layer_name]["Bottom_weights"] = self.get_weights_layer_bottom(weights_layer_name, weights_layer_bottom=[])

            if Weights_layer_dict[weights_layer_name].type =='Convolution':
                Weights_layers_graph[weights_layer_name]["output_num"] = Weights_layer_dict[weights_layer_name].convolution_param.num_output
                Weights_layers_graph[weights_layer_name]["type"] = "Convolution"

            else:
                Weights_layers_graph[weights_layer_name]["output_num"] = Weights_layer_dict[weights_layer_name].inner_product_param.num_output
                Weights_layers_graph[weights_layer_name]["type"] = "InnerProduct"

        return Weights_layers_graph

    # todo: 返回需要砍相同channel的layer组成的set， 更将各个set按照先后顺序放在list里面：
    def get_weights_pruning_group(self, order=1):
        weights_pruning_group = []
        weights_pruning_tmp = []
        all_weights_set = set()
        for weights_name in self.weights_layers_graph.keys()[::-1]:
            bottom_weights = self.weights_layers_graph[weights_name]["Bottom_weights"]
            if len(bottom_weights) == 0:
                continue
            if_new_set = True
            for bottom_i in bottom_weights:
                if bottom_i in all_weights_set:
                    if_new_set = False
            if (if_new_set):
                all_weights_set = all_weights_set | set(bottom_weights)
                weights_pruning_tmp.append(bottom_weights)

        keys_ = self.weights_layers_graph.keys()
        if(order!=1):
            keys_ = keys_[::-1]
        for weights_name in keys_:
            for set_i in weights_pruning_tmp:
                if weights_name in set_i:
                    weights_pruning_group.append(set_i)
                    weights_pruning_tmp.remove(set_i)
        return weights_pruning_group

    # todo:
    # （1）当前层的名字 name (ok)
    # （2）当前层的type  (ok)
    # （3）直接连接的top_layers  (ok)
    # （4）直接连接的bottom_layers  (ok)
    def get_layer_relation(self):
        print("TODO")

def Get_pruning_step_group(model_file, ignore_layers = None, order=1):
    Net_composition = Net_composition_(model_file, order=order)
    weights_layers_graph = Net_composition.weights_layers_graph
    weights_pruning_group = Net_composition.weights_pruning_group

    pruning_step_group = []
    for pruning_group_layers_i in weights_pruning_group:
        if not ignore_layers is None:
            if len(set(pruning_group_layers_i) & set(ignore_layers)) != 0:
                continue
        pruning_step_gingle = OrderedDict()

        for layer_i in pruning_group_layers_i:
            step_num = int(weights_layers_graph[layer_i]["output_num"] * cfg.C_PRUNE.STEP_RATIO)
            pruning_step_gingle[layer_i] = step_num
        pruning_step_group.append(pruning_step_gingle)
    return pruning_step_group

def Get_pruning_singlelayer_step(model_file, layers_list):
    Net_composition = Net_composition_(model_file)
    weights_layers_graph = Net_composition.weights_layers_graph

    pruning_step_gingle = OrderedDict()
    for layer_i in layers_list:
        step_num = math.ceil(weights_layers_graph[layer_i]["output_num"] * cfg.C_PRUNE.STEP_RATIO)
        if step_num <= cfg.C_PRUNE.STEP_MIN:
            step_num = cfg.C_PRUNE.STEP_MIN
        if step_num >= weights_layers_graph[layer_i]["output_num"] :
            step_num = 0
        pruning_step_gingle[layer_i] = step_num

    return pruning_step_gingle


class Prune_parameters_():
    def __init__(self, args):
        self.iter = 0
        self.args = args
        self.model_file = os.path.join(self.args.task_root, "{}.prototxt".format(self.args.model_file))
        self.new_model_file = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.model_file, self.iter))
        self.model_file_train = os.path.join(self.args.task_root, "{}.prototxt".format(self.args.model_file_train))
        self.new_model_file_train = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.model_file_train, self.iter))
        self.weights_file = os.path.join(self.args.task_root, "{}.caffemodel".format(self.args.weights_file))
        self.new_weights_file = os.path.join(self.args.task_root, "{}_{}.caffemodel".format(self.args.weights_file, self.iter))

        self.solver_file = os.path.join(self.args.task_root, "{}.prototxt".format(self.args.solver_file))
        self.new_solver_file = os.path.join(self.args.task_root,
                                             "{}_{}.prototxt".format(self.args.solver_file, self.iter))
        self.GPU_index = self.args.gpu_id

    def update_iter(self, new_iter):
        self.iter = new_iter
        return self.iter

    def update_file(self, new_iter):
        self.model_file = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.model_file, self.iter))
        self.model_file_train = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.model_file_train, self.iter))
        self.weights_file = os.path.join(self.args.task_root, "{}_{}.caffemodel".format(self.args.weights_file, self.iter))
        self.solver_file = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.solver_file, self.iter))

        self.update_iter(new_iter)

        self.new_model_file = os.path.join(self.args.task_root, "{}_{}.prototxt".format(self.args.model_file, self.iter))
        self.new_model_file_train = os.path.join(self.args.task_root,
                                                 "{}_{}.prototxt".format(self.args.model_file_train, self.iter))
        self.new_weights_file = os.path.join(self.args.task_root, "{}_{}.caffemodel".format(self.args.weights_file, self.iter))
        self.new_solver_file = os.path.join(self.args.task_root,
                                            "{}_{}.prototxt".format(self.args.solver_file, self.iter))

class Prune_tools():
    # P_parameters为Prune_parameters，保存了剪枝相关参数
    def __init__(self, P_parameters):

        # get ori net weights
        self.P_parameters = P_parameters
        self.ori_net = load_net_(self.P_parameters.model_file, weights_file = self.P_parameters.weights_file,
                                GPU_index = self.P_parameters.GPU_index, batch_size = -1, forward_type = "test")

        self.Net_composition = Net_composition_(self.P_parameters.model_file)

        self.pruned_layers_set = set()
        self.fixed_layers_set = set()

    def set_device(self, GPU_index =None):
        if GPU_index ==None:
            caffe.set_device(self.P_parameters.GPU_index)
        else:
            caffe.set_device(GPU_index)

    # todo：实现新的model相对的prototxt的书写
    def save_new_model_file(self, model_file, new_model_file, pruning_dict):
        # 解析原始模型
        with open(model_file) as f:
            model = caffe.proto.caffe_pb2.NetParameter()
            pb.text_format.Merge(f.read(), model)

        for i, layer in enumerate(model.layer):
            if layer.name in pruning_dict.keys():
                if layer.type == 'Convolution':
                    layer.convolution_param.num_output = int(layer.convolution_param.num_output - pruning_dict[layer.name])

                if layer.type == 'InnerProduct':
                    layer.inner_product_param.num_output = int(layer.inner_product_param.num_output - pruning_dict[layer.name])

        # 保存新模型prototxt
        with open(new_model_file, 'w') as f:
            f.write(pb.text_format.MessageToString(model))
        print("Save new proto file: {}".format(new_model_file))

    def save_new_solver_file(self, solver_file, new_solver_file, new_model_file):
        # 解析原始模型
        with open(solver_file) as f:
            solver_ = caffe.proto.caffe_pb2.SolverParameter()
            pb.text_format.Merge(f.read(), solver_)
        solver_.net = new_model_file
        solver_.snapshot_prefix = os.path.join(os.path.dirname(solver_.snapshot_prefix),os.path.basename(new_model_file.split(".prototxt")[0]))

        # 保存新模型prototxt
        with open(new_solver_file, 'w') as f:
            f.write(pb.text_format.MessageToString(solver_))
        print("Save new proto file: {}".format(new_solver_file))

    def update_model_file(self, pruning_dict):
        self.save_new_model_file(self.P_parameters.model_file, self.P_parameters.new_model_file, pruning_dict)
        self.save_new_model_file(self.P_parameters.model_file_train, self.P_parameters.new_model_file_train, pruning_dict)
        self.save_new_solver_file(self.P_parameters.solver_file, self.P_parameters.new_solver_file, self.P_parameters.new_model_file_train)

    def create_new_net(self):
        self.new_net = load_net_(self.P_parameters.new_model_file, weights_file = None,
                                GPU_index = self.P_parameters.GPU_index, batch_size = -1, forward_type = "test")

    # fill BN weights
    def Get_pruned_BN(self, BN_name, keep_index_list_BN):
        if len(keep_index_list_BN) == 0:
            keep_index_list_BN = range(len(self.ori_net.params[BN_name][0].data))
        self.new_net.params[BN_name][0].data[:] = self.ori_net.params[BN_name][0].data[keep_index_list_BN]
        self.new_net.params[BN_name][1].data[:] = self.ori_net.params[BN_name][1].data[keep_index_list_BN]
        self.new_net.params[BN_name][2].data[:] = self.ori_net.params[BN_name][2].data
        # return self.new_net

    # fill scale weights
    def Get_pruned_scale(self, scale_name, keep_index_list_scale):
        if len(keep_index_list_scale) == 0:
            keep_index_list_scale = range(len(self.ori_net.params[scale_name][0].data))
        self.new_net.params[scale_name][0].data[:] = self.ori_net.params[scale_name][0].data[keep_index_list_scale]
        self.new_net.params[scale_name][1].data[:] = self.ori_net.params[scale_name][1].data[keep_index_list_scale]

        # return self.new_net

    # fill conv weights
    def Get_pruned_conv(self, conv_name, Convolution_pruning_dict):
        keep_index_list_conv_ch1 = Convolution_pruning_dict["ch_1"] if len(
            Convolution_pruning_dict["ch_1"]) > 0 else range(len(self.ori_net.params[conv_name][0].data))
        keep_index_list_conv_ch2 = Convolution_pruning_dict["ch_2"] if len(
            Convolution_pruning_dict["ch_2"]) > 0 else range(len(self.ori_net.params[conv_name][0].data[0]))
        a = self.ori_net.params[conv_name][0].data[keep_index_list_conv_ch1, :, :, :]
        b = a[:, keep_index_list_conv_ch2, :, :]
        self.new_net.params[conv_name][0].data[...] = b
        if len(self.ori_net.params[conv_name]) > 1:
            self.new_net.params[conv_name][1].data[...] = self.ori_net.params[conv_name][1].data[keep_index_list_conv_ch1]
        # return self.new_net

    # fill fc weights
    def Get_pruned_fc(self, fc_name, InnerProduct_pruning_dict):
        keep_index_list_fc_ch1 = InnerProduct_pruning_dict["ch_1"] if len(
            InnerProduct_pruning_dict["ch_1"]) > 0 else range(len(self.ori_net.params[fc_name][0].data))
        keep_index_list_fc_ch2 = InnerProduct_pruning_dict["ch_2"] if len(
            InnerProduct_pruning_dict["ch_2"]) > 0 else range(len(self.ori_net.params[fc_name][0].data[0]))
        a = self.ori_net.params[fc_name][0].data[keep_index_list_fc_ch1, :]
        b = a[:, keep_index_list_fc_ch2]

        self.new_net.params[fc_name][0].data[...] = b
        if len(self.ori_net.params[fc_name]) > 1:
            self.new_net.params[fc_name][1].data[...] = self.ori_net.params[fc_name][1].data[keep_index_list_fc_ch1]
        # return self.new_net

    def Get_pruned_other_weights(self, weights_name):
        for index_i in range(len(self.ori_net.params[weights_name])):
            self.new_net.params[weights_name][index_i].data[...] = self.ori_net.params[weights_name][index_i].data
        # return self.new_net

    def update_new_net(self, pruning_keep_index_dict):
        for layer_i in self.new_net.params:
            if layer_i in pruning_keep_index_dict:
                if self.Net_composition.all_layer_graph[layer_i].type == "BatchNorm":
                    self.Get_pruned_BN(layer_i, pruning_keep_index_dict[layer_i])

                if self.Net_composition.all_layer_graph[layer_i].type == "Scale":
                    self.Get_pruned_scale(layer_i, pruning_keep_index_dict[layer_i])

                if self.Net_composition.all_layer_graph[layer_i].type == "Convolution":
                    self.Get_pruned_conv(layer_i, pruning_keep_index_dict[layer_i])

                if self.Net_composition.all_layer_graph[layer_i].type == "InnerProduct":
                    self.Get_pruned_fc(layer_i, pruning_keep_index_dict[layer_i])
            else:
                self.Get_pruned_other_weights(layer_i)
        # return self.new_net

    # todo：保存更新后的model参数
    def save_new_weights(self):
        self.new_net.save(self.P_parameters.new_weights_file)

    def save_pruning(self, last_model_path, IF_PRUNING = True):
        if not IF_PRUNING:
            pruning_dict = {}
            self.new_net = self.ori_net
            self.update_model_file(pruning_dict)
            self.save_new_weights()
        else:
            #train_net.save(self.P_parameters.new_weights_file)
            shutil.copy(last_model_path, self.P_parameters.new_weights_file)
            #self.new_net = train_net
        

def single_batch_forward(net, start = None, end = None):
    # net.blobs['data'].data[...] = np.array(single_input_data)
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

def pruned_channel_forward_by_scale(net, weights_layers_graph, weights_layer, pruning_num):
    Scale_name = weights_layers_graph[weights_layer]["Scale"]
    feature_loss = []

    scale_tmp_0 = 0
    scale_tmp_1 = 0
    for i in range(weights_layers_graph[weights_layer]["output_num"]):
        # print("Test channel {}.".format(i))
        # print
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

        out_put, net = single_batch_forward(net, end = "loss")
        feature_loss.append(copy.deepcopy(out_put[out_put.keys()[-1]]))
        # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))
        # print("feature: {}".format(out_put[out_put.keys()[-1]]))

    net.params[Scale_name][0].data[weights_layers_graph[weights_layer]["output_num"] - 1] = scale_tmp_0
    net.params[Scale_name][1].data[weights_layers_graph[weights_layer]["output_num"] - 1] = scale_tmp_1
    sort_index = np.argsort(feature_loss)
    # print sort_index
    # print feature_loss
    # print sort_index[0:pruning_num]
    return sort_index[0:pruning_num]


def get_BN_index_scale(net, weights_layers_graph, pruning_layer_dict):
    channel_num = len(net.params[weights_layers_graph[pruning_layer_dict.keys()[0]]["Scale"]][0].data[...])
    step_size = pruning_layer_dict[pruning_layer_dict.keys()[0]]
    channel_num_new = int(channel_num - step_size)

    feature_loss = []

    scale_tmp_0 = [0 for i in range(len(pruning_layer_dict.keys()))]
    scale_tmp_1 = [0 for i in range(len(pruning_layer_dict.keys()))]
    for i in range(channel_num):
        # print("Test channel {}.".format(i))
        # print
        for layer_index, weights_name in enumerate(pruning_layer_dict.keys()):
            Scale_name = weights_layers_graph[weights_name]["Scale"]
            if i != 0:
                net.params[Scale_name][0].data[i - 1] = scale_tmp_0[layer_index]
                net.params[Scale_name][1].data[i - 1] = scale_tmp_1[layer_index]
            # print("============================================")
            # print("scale: {}.".format(net.params[Scale_name][0].data[...]))
            scale_tmp_0[layer_index] = net.params[Scale_name][0].data[i]
            scale_tmp_1[layer_index] = net.params[Scale_name][1].data[i]
            net.params[Scale_name][0].data[i] = 0
            net.params[Scale_name][1].data[i] = 0
            # print("scale: {}.".format(net.params[Scale_name][0].data[...]))

        out_put, net = single_batch_forward(net, end="loss")
        feature_loss.append(copy.deepcopy(out_put[out_put.keys()[-1]]))
            # print("feature: {}".format(out_put[out_put.keys()[-1]][0:3]))
        logging.error("feature: {}".format(out_put[out_put.keys()[-1]]))
    for layer_index, weights_name in enumerate(pruning_layer_dict.keys()):
        Scale_name = weights_layers_graph[weights_name]["Scale"]
        net.params[Scale_name][0].data[channel_num - 1] = scale_tmp_0[layer_index]
        net.params[Scale_name][1].data[channel_num - 1] = scale_tmp_1[layer_index]
    sort_index_by_scale = np.argsort(feature_loss)
    # for ss in range(len(sort_index_by_scale)):
    #     logging.error("sort index: {}".format(sort_index_by_scale[ss]))
    #     logging.error("sort feature: {}".format(feature_loss[sort_index_by_scale[ss]]))
    # logging.error("sort: {}".format(" ".join(["{}".format(ss) for ss in feature_loss[sort_index_by_scale]])))

    BN_keep_index = OrderedDict()
    # sort_index = np.argsort(sort_index_by_scale)
    keep_index_i = sort_index_by_scale[-channel_num_new:].tolist()
    keep_index_i = np.sort(keep_index_i)
    for ss in range(len(keep_index_i)):
        logging.error("sort index: {}".format(keep_index_i[ss]))
        logging.error("sort feature: {}".format(feature_loss[keep_index_i[ss]]))
    for weights_name in pruning_layer_dict.keys():
        bn_name = weights_layers_graph[weights_name]["BN"]
        BN_keep_index[bn_name] = keep_index_i

    return BN_keep_index


##  搜寻pruning channel 的index 方法一：根据BN参数搜索
def get_BN_index(net, weights_layers_graph, pruning_layer_dict):
    layer_data_0 = 0
    layer_data_1 = 0
    layer_num = len(pruning_layer_dict.keys())
    BN_keep_index = OrderedDict()
    for weights_name in pruning_layer_dict.keys():
        step_size = pruning_layer_dict[weights_name]
        bn_name = weights_layers_graph[weights_name]["BN"]
        layer_data_0 += net.params[bn_name][0].data
        layer_data_1 *= net.params[bn_name][1].data

    channel_num_new = int(len(layer_data_0) - step_size)
    if channel_num_new <1:
        channel_num_new =1
    layer_check = np.abs(layer_data_0 / layer_num) * (layer_data_1**(1/layer_num))

    sort_index = np.argsort(layer_check)
    keep_index_i = sort_index[:channel_num_new].tolist()
    keep_index_i = np.sort(keep_index_i)
    for weights_name in pruning_layer_dict.keys():
        bn_name = weights_layers_graph[weights_name]["BN"]
        BN_keep_index[bn_name] = keep_index_i
    return BN_keep_index

def get_all_keep_index(BN_pruning_dict, pruning_layer_dict, weights_graph):
    Weights_pruning_dict = {}

    for pruning_layer_i in pruning_layer_dict.keys():
        BN_layer_name = weights_graph[pruning_layer_i]["BN"]
        keep_index_list_BN = BN_pruning_dict[BN_layer_name]
        Weights_pruning_dict[BN_layer_name] = keep_index_list_BN

        Scale_layer_name = weights_graph[pruning_layer_i]["Scale"]
        Weights_pruning_dict[Scale_layer_name] = keep_index_list_BN


        if weights_graph[pruning_layer_i]["type"] == 'Convolution':
            if not pruning_layer_i in Weights_pruning_dict:
                Weights_pruning_dict[pruning_layer_i] = {}
                Weights_pruning_dict[pruning_layer_i]["ch_1"] = []
                Weights_pruning_dict[pruning_layer_i]["ch_2"] = []
            Weights_pruning_dict[pruning_layer_i]["ch_1"] = keep_index_list_BN

            for weights_top_i in weights_graph[pruning_layer_i]["Top_weights"]:
                if weights_graph[weights_top_i]["type"] == 'Convolution':
                    if not weights_top_i in Weights_pruning_dict:
                        Weights_pruning_dict[weights_top_i] = {}
                        Weights_pruning_dict[weights_top_i]["ch_1"] = []
                        Weights_pruning_dict[weights_top_i]["ch_2"] = []
                    Weights_pruning_dict[weights_top_i]["ch_2"] = keep_index_list_BN

                elif weights_graph[weights_top_i]["type"] == 'InnerProduct':
                    if not weights_top_i in Weights_pruning_dict:
                        Weights_pruning_dict[weights_top_i] = {}
                        Weights_pruning_dict[weights_top_i]["ch_1"] = []
                        Weights_pruning_dict[weights_top_i]["ch_2"] = []
                    Weights_pruning_dict[weights_top_i]["ch_2"] = keep_index_list_BN

        if weights_graph[pruning_layer_i]["type"] == 'InnerProduct':
            if not pruning_layer_i in Weights_pruning_dict:
                Weights_pruning_dict[pruning_layer_i] = {}
                Weights_pruning_dict[pruning_layer_i]["ch_1"] = []
                Weights_pruning_dict[pruning_layer_i]["ch_2"] = []
            Weights_pruning_dict[pruning_layer_i]["ch_1"] = keep_index_list_BN

            for weights_top_i in weights_graph[pruning_layer_i]["Top_weights"]:
                if weights_graph[weights_top_i]["type"] == 'Convolution':
                    if not weights_top_i in Weights_pruning_dict:
                        Weights_pruning_dict[weights_top_i] = {}
                        Weights_pruning_dict[weights_top_i]["ch_1"] = []
                        Weights_pruning_dict[weights_top_i]["ch_2"] = []
                    Weights_pruning_dict[weights_top_i]["ch_2"] = keep_index_list_BN

                elif weights_graph[weights_top_i]["type"] == 'InnerProduct':
                    if not weights_top_i in Weights_pruning_dict:
                        Weights_pruning_dict[weights_top_i] = {}
                        Weights_pruning_dict[weights_top_i]["ch_1"] = []
                        Weights_pruning_dict[weights_top_i]["ch_2"] = []
                    Weights_pruning_dict[weights_top_i]["ch_2"] = keep_index_list_BN

    return Weights_pruning_dict


if __name__ == '__main__':
    print("todo")



