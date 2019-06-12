#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
from config import cfg, cfg_from_file, cfg_from_list
import pprint
import time, os, sys
import numpy as np
import cv2

def load_data_batch_with_label(net, img_root, img_list):
    # print("cfg.IMG_TYPE: {}".format(cfg.IMG_TYPE))
    input_layer_name = net._layer_names[net._inputs[0]]
    b, c, h, w = net.blobs[input_layer_name].shape

    input_data = []
    input_label = []
    input_data_name = []
    for index, img_path_i in enumerate(img_list):
        img_path_label = img_path_i.split("\n")[0]
        img_label = img_path_label.split(" ")[-1]
        img_name = " ".join(img_path_label.split(" ")[:-1])

        img_path = os.path.join(img_root, img_name)

        if not (os.path.isfile(img_path) and os.path.splitext(img_path)[-1] in [".jpg", ".png", ".bmp"]):
            continue

        # opencv读取数据
        if c == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 彩色图
        img_h, img_w, img_c = image.shape

        if cfg.IMG_TYPE == "resize":
            # 若尺寸不对(例如使用原图)进行resize
            if img_h != cfg.IMG_RESIZE_SIZE_H or img_w != cfg.IMG_RESIZE_SIZE_W:
                # image = cv2.resize(image, (resize_size_w,resize_size_h), interpolation=cv2.INTER_AREA)
                img_roi = cv2.resize(image, (cfg.IMG_RESIZE_SIZE_W, cfg.IMG_RESIZE_SIZE_H), interpolation=cv2.INTER_LINEAR)

        elif cfg.IMG_TYPE == "crop":
            # 若crop尺寸小于例如使用原图)进行resize
            if img_h >= cfg.IMG_CROP_SIZE_H:
                start_h = int((img_h - cfg.IMG_CROP_SIZE_H) / 2)
                crop_h = cfg.IMG_CROP_SIZE_H
            else:
                start_h = 0
                crop_h = img_h

            if img_w >= cfg.IMG_CROP_SIZE_W:
                start_w = int((img_w - cfg.IMG_CROP_SIZE_W) / 2)
                crop_w = cfg.IMG_CROP_SIZE_W
            else:
                start_w= 0
                crop_w = img_w

            img_roi = image[start_h: start_h + crop_h, start_w: start_w + crop_w]

        elif cfg.IMG_TYPE == "resize_crop":
            if img_h != cfg.IMG_RESIZE_SIZE_H or img_w != cfg.IMG_RESIZE_SIZE_W:
                # image = cv2.resize(image, (resize_size_w,resize_size_h), interpolation=cv2.INTER_AREA)
                image = cv2.resize(image, (cfg.IMG_RESIZE_SIZE_W, cfg.IMG_RESIZE_SIZE_H), interpolation=cv2.INTER_LINEAR)

            # 若crop尺寸小于例如使用原图)进行resize
            if cfg.IMG_RESIZE_SIZE_H >= cfg.IMG_CROP_SIZE_H:
                start_h = int((cfg.IMG_RESIZE_SIZE_H - cfg.IMG_CROP_SIZE_H) / 2)
                crop_h = cfg.IMG_CROP_SIZE_H
            else:
                start_h = 0
                crop_h = cfg.IMG_RESIZE_SIZE_H

            if cfg.IMG_RESIZE_SIZE_W >= cfg.IMG_CROP_SIZE_W:
                start_w = int((cfg.IMG_RESIZE_SIZE_W - cfg.IMG_CROP_SIZE_W) / 2)
                crop_w = cfg.IMG_CROP_SIZE_W
            else:
                start_w = 0
                crop_w = cfg.IMG_RESIZE_SIZE_W

            img_roi = image[start_h: start_h + crop_h, start_w: start_w + crop_w]

        img_roi_h, img_roi_w, img_roi_c = img_roi.shape
        if(img_roi_h != h or img_roi_w != w):
            img_roi = cv2.resize(img_roi, (w, h), interpolation=cv2.INTER_LINEAR)

        if c == 1:
            img = np.empty(shape=[1, h, w])  # img = np.zeros((1,3,96,96))
            if cfg.IF_INPUT_MEANS:
                img[0, :, :] = img_roi[:, :, 0] - cfg.PIXEL_MEANS[0, 0 ,0]  # img_roi[:,:,0] - 104.0;
            else:
                img[0, :, :] = img_roi[:, :, 0]
        else:
            img = np.empty(shape=[3, h, w])  # img = np.zeros((1,3,96,96))
            if cfg.IF_INPUT_MEANS:
                img[0, :, :] = img_roi[:, :, 0] - cfg.PIXEL_MEANS[0, 0, 0]  # img_roi[:,:,0] - 104.0;
                img[1, :, :] = img_roi[:, :, 1] - cfg.PIXEL_MEANS[0, 0, 1]  # img_roi[:,:,1] - 113.0;
                img[2, :, :] = img_roi[:, :, 2] - cfg.PIXEL_MEANS[0, 0, 2]  # img_roi[:,:,2] - 127.0;
            else:
                img[0, :, :] = img_roi[:, :, 0]
                img[1, :, :] = img_roi[:, :, 1]
                img[2, :, :] = img_roi[:, :, 2]


        input_data.append(img)
        input_label.append(img_label)
        # input_data_name.append(os.path.basename(img_name))
        input_data_name.append(img_path)

    return input_data, input_label, input_data_name


def load_data_batch(net, img_root, img_list):
    input_layer_name = net._layer_names[net._inputs[0]]
    b, c, h, w = net.blobs[input_layer_name].shape

    input_data = []
    for index, img_path_i in enumerate(img_list):
        img_name = img_path_i.split("\n")[0]

        img_path = os.path.join(img_root, img_name)

        if not (os.path.isfile(img_path) and os.path.splitext(img_path)[-1] in [".jpg", ".png", ".bmp"]):
            continue

        # opencv读取数据
        if c == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 彩色图
        img_h, img_w, img_c = image.shape

        if cfg.IMG_TYPE == "resize":
            # 若尺寸不对(例如使用原图)进行resize
            if img_h != cfg.IMG_RESIZE_SIZE_H or img_w != cfg.IMG_RESIZE_SIZE_W:
                # image = cv2.resize(image, (resize_size_w,resize_size_h), interpolation=cv2.INTER_AREA)
                img_roi = cv2.resize(image, (cfg.IMG_RESIZE_SIZE_W, cfg.IMG_RESIZE_SIZE_H), interpolation=cv2.INTER_LINEAR)

        elif cfg.IMG_TYPE == "crop":
            # 若crop尺寸小于例如使用原图)进行resize
            if img_h >= cfg.IMG_CROP_SIZE_H:
                start_h = int((img_h - cfg.IMG_CROP_SIZE_H) / 2)
                crop_h = cfg.IMG_CROP_SIZE_H
            else:
                start_h = 0
                crop_h = img_h

            if img_w >= cfg.IMG_CROP_SIZE_W:
                start_w = int((img_w - cfg.IMG_CROP_SIZE_W) / 2)
                crop_w = cfg.IMG_CROP_SIZE_W
            else:
                start_w= 0
                crop_w = img_w

            img_roi = image[start_h: start_h + crop_h, start_w: start_w + crop_w]

        elif cfg.IMG_TYPE == "resize_crop":
            if img_h != cfg.IMG_RESIZE_SIZE_H or img_w != cfg.IMG_RESIZE_SIZE_W:
                # image = cv2.resize(image, (resize_size_w,resize_size_h), interpolation=cv2.INTER_AREA)
                image = cv2.resize(image, (cfg.IMG_RESIZE_SIZE_W, cfg.IMG_RESIZE_SIZE_H), interpolation=cv2.INTER_LINEAR)

            # 若crop尺寸小于例如使用原图)进行resize
            if cfg.IMG_RESIZE_SIZE_H >= cfg.IMG_CROP_SIZE_H:
                start_h = int((cfg.IMG_RESIZE_SIZE_H - cfg.IMG_CROP_SIZE_H) / 2)
                crop_h = cfg.IMG_CROP_SIZE_H
            else:
                start_h = 0
                crop_h = cfg.IMG_RESIZE_SIZE_H

            if cfg.IMG_RESIZE_SIZE_W >= cfg.IMG_CROP_SIZE_W:
                start_w = int((cfg.IMG_RESIZE_SIZE_W - cfg.IMG_CROP_SIZE_W) / 2)
                crop_w = cfg.IMG_CROP_SIZE_W
            else:
                start_w = 0
                crop_w = cfg.IMG_RESIZE_SIZE_W

            img_roi = image[start_h: start_h + crop_h, start_w: start_w + crop_w]

        img_roi_h, img_roi_w, img_roi_c = img_roi.shape
        if(img_roi_h != h or img_roi_w != w):
            img_roi = cv2.resize(img_roi, (w, h), interpolation=cv2.INTER_LINEAR)

        if c == 1:
            img = np.empty(shape=[1, h, w])  # img = np.zeros((1,3,96,96))
            if cfg.IF_INPUT_MEANS:
                img[0, :, :] = img_roi[:, :, 0] - cfg.PIXEL_MEANS[0, 0 ,0]  # img_roi[:,:,0] - 104.0;
            else:
                img[0, :, :] = img_roi[:, :, 0]
        else:
            img = np.empty(shape=[3, h, w])  # img = np.zeros((1,3,96,96))
            if cfg.IF_INPUT_MEANS:
                img[0, :, :] = img_roi[:, :, 0] - cfg.PIXEL_MEANS[0, 0, 0]  # img_roi[:,:,0] - 104.0;
                img[1, :, :] = img_roi[:, :, 1] - cfg.PIXEL_MEANS[0, 0, 1]  # img_roi[:,:,1] - 113.0;
                img[2, :, :] = img_roi[:, :, 2] - cfg.PIXEL_MEANS[0, 0, 2]  # img_roi[:,:,2] - 127.0;
            else:
                img[0, :, :] = img_roi[:, :, 0]
                img[1, :, :] = img_roi[:, :, 1]
                img[2, :, :] = img_roi[:, :, 2]

        input_data.append(img)

    return input_data


