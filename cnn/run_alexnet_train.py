#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
import argparse
from alexnet.train import train_alexnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="When True, runs a modified training script",
                        type=bool, default=False)
    args = parser.parse_args()
    train_alexnet(debug=args.debug)

if __name__ == '__main__':
    main()