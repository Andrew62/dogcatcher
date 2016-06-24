#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
import argparse
from config import workspace
from trainer import train_model
from vgg.vgg16_C import VGG16_C

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="When True, runs a modified training script",
                        type=bool, default=True)
    args = parser.parse_args()
    train_model(workspace.class_pkl, workspace.train_pkl, VGG16_C, workspace.vgg_models, debug=args.debug)

if __name__ == '__main__':
    main()