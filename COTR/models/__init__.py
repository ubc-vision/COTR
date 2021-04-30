'''
The COTR model is modified from DETR code base.
https://github.com/facebookresearch/detr
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .cotr_model import build


def build_model(args):
    return build(args)
