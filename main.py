#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from Segmentor import Segmentor


def parse():
    parser = argparse.ArgumentParser(description='Chinese Words Segment')
    parser.add_argument(
        '--more',
        type=int,
        default=0,
        help='Use large training corpus or not, default by 1')
    parser.add_argument(
        '--save',
        type=int,
        default=1,
        help='Save the term frequency dictionary or not, default by 1')
    parser.add_argument(
        '--method',
        type=str,
        default='MP',
        help='Choose the method to segment:HMM,FMM,MP, default by MP')
    parser.add_argument(
        '--path',
        type=str,
        default='./test.txt',
        help='Path of the file to be segmented, default by "./test.txt"')

    args = parser.parse_args()
    return args


def run(args):
    more, save, method, path = args.more, args.save, args.method, args.path
    segmentor = Segmentor(more)
    segmentor.build_dic(save)
    segmentor.load_dict()
    segmentor.seg(method, path)
    # dic_uni = segmentor.get_dict(1)
    # dic_bi = segmentor.get_dict(2)


if __name__ == '__main__':
    args = parse()
    run(args)
