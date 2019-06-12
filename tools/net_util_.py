#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
__author__ = 'yxh'

import numpy as np
import cv2
import time

from collections import OrderedDict

import caffe


def load_net_scale(model_file, weights_file = None, GPU_index = -1, forward_type = "test"):
    if GPU_index != -1:
        caffe.set_device(GPU_index)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    if weights_file == None:
        if forward_type == "train":
            net = caffe.Net(model_file, caffe.TRAIN)
        else:
            net = caffe.Net(model_file, caffe.TEST)
    else:
        if forward_type == "train":
            net = caffe.Net(model_file, weights_file, caffe.TRAIN)
        else:
            net = caffe.Net(model_file, weights_file, caffe.TEST)

    return net

def load_net_(model_file, weights_file = None, GPU_index = -1, batch_size = -1, forward_type = "test"):
    if GPU_index != -1:
        caffe.set_device(GPU_index)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    if weights_file == None:
        if forward_type == "train":
            net = caffe.Net(model_file, caffe.TRAIN)
        else:
            net = caffe.Net(model_file, caffe.TEST)
    else:
        if forward_type == "train":
            net = caffe.Net(model_file, weights_file, caffe.TRAIN)
        else:
            net = caffe.Net(model_file, weights_file, caffe.TEST)

    # input_layer_name = net._layer_names[net._inputs[0]]
    # b, c, h, w = net.blobs[input_layer_name].shape
    # if batch_size != -1:
    #     net.blobs[input_layer_name].reshape(batch_size, c, h, w)
    return net

def change_net_batchsize(net, batch_size):
    input_layer_name = net._layer_names[net._inputs[0]]
    b, c, h, w = net.blobs[input_layer_name].shape
    net.blobs[input_layer_name].reshape(batch_size, c, h, w)
    return net

def predict_with_label(net, input_data, input_label, input_data_name):
    predict_res = []
    batch_size, c, h, w = net.blobs[net._layer_names[net._inputs[0]]].data.shape
    img_len = len(input_data)
    epoch_num = int(img_len/batch_size)
    # print (epoch_num)
    for epoch_i in range(epoch_num):
        # if epoch_i % 10 == 0:
        #     print ("TODO: {}%: ".format(1.0*epoch_i/epoch_num*100))
        epoch_len = len(input_data[epoch_i*batch_size:(epoch_i + 1)*batch_size])
        img = np.zeros((batch_size, c, h, w))

        img[0:epoch_len] = input_data[epoch_i*batch_size:(epoch_i + 1)*batch_size]

        net.blobs['data'].data[...] = img

        ### 执行分类
        # time_0 = time.time()
        output = net.forward()
        # time_1 = time.time()
        # print ("time: {}".format(time_1 - time_0))
        output_prob = output['prob']  # batch中第一张图像的概率值
        for idx, idx_prob in enumerate(output_prob[:epoch_len]):
            single_pred = []
            for score in idx_prob:
                single_pred.append(score)
            single_res = {}
            single_res["name"] = input_data_name[idx + epoch_i*batch_size]
            single_res["label"] = input_label[idx + epoch_i * batch_size]
            single_res["pred"] = single_pred
            predict_res.append(single_res)
    return predict_res

def predict(net, input_data):
    predict_res = []
    batch_size, c, h, w = net.blobs[net._layer_names[net._inputs[0]]].data.shape
    img_len = len(input_data)
    epoch_num = int(img_len/batch_size)
    # print (epoch_num)
    for epoch_i in range(epoch_num):
        # if epoch_i % 10 == 0:
        #     print ("TODO: {}%: ".format(1.0*epoch_i/epoch_num*100))
        epoch_len = len(input_data[epoch_i*batch_size:(epoch_i + 1)*batch_size])
        img = np.zeros((batch_size, c, h, w))

        img[0:epoch_len] = input_data[epoch_i*batch_size:(epoch_i + 1)*batch_size]

        net.blobs['data'].data[...] = img

        ### 执行分类
        # time_0 = time.time()
        output = net.forward()
        # time_1 = time.time()
        # print ("time: {}".format(time_1 - time_0))
        output_prob = output['prob']  # batch中第一张图像的概率值
        for idx, idx_prob in enumerate(output_prob[:epoch_len]):
            single_pred = []
            for score in idx_prob:
                single_pred.append(score)
            single_res = {}
            single_res["pred"] = single_pred
            predict_res.append(single_res)
    return predict_res

if __name__ == '__main__':
    model_file = "./res18-deploy-class2.prototxt"
    weights_file = "./test_whole_checkbox_resize_singlelabel_dty_side_jilian_20190403_iter_191000.caffemodel"
    net = load_net_(model_file, weights_file=weights_file, GPU_index=-1, batch_size=-1, forward_type="test")

    input_layer_name = net._layer_names[net._inputs[0]]
    b, c, h, w = net.blobs[input_layer_name].shape
    print("Batch size: {}".format(b))

    net = change_net_batchsize(net, 12)
    b, c, h, w = net.blobs[input_layer_name].shape
    print("Batch size: {}".format(b))





