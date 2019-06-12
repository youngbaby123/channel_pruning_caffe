#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)
    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs)
    #
    # cfg.GPU_ID = args.gpu_id
    #
    # print('Using config:')
    # pprint.pprint(cfg)
    #
    #
    # caffe.set_device(args.gpu_id)
    #
    # if not args.randomize:
    #     # fix the random seeds (numpy and caffe) for reproducibility
    #     np.random.seed(cfg.RNG_SEED)
    #     caffe.set_random_seed(cfg.RNG_SEED)
    #
    # # set up caffe
    # caffe.set_mode_gpu()
    #
    # imdb, roidb = combined_roidb(args.imdb_name)
    # #for item in roidb:
    # #  print item
    # #  raw_input()
    # print '{:d} roidb entries'.format(len(roidb))
    #
    # output_dir = get_output_dir(imdb)
    # print 'Output will be saved to `{:s}`'.format(output_dir)
    #
    # train_net(args.solver, roidb, output_dir,
    #           pretrained_model=args.pretrained_model,
    #           max_iters=args.max_iters)
