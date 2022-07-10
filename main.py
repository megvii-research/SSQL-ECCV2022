#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
from workers import default, ptq, ssl, draw, linear 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("SSQL ECCV2022")
    subparsers = parser.add_subparsers(help="sub-command help")

    # a worker train float model
    parser_default = subparsers.add_parser(
        "default", help="the entrance of Float Model Training"
    )
    parser_default.set_defaults(func=default)
    parser_default.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_default.add_argument(
        "-j", "--num_workers", type=int, required=False, default=4
    )
    parser_default.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_default.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

     # a worker for linear evaluation
    parser_linear = subparsers.add_parser(
        "linear", help="the entrance of Linear Evaluation"
    )
    parser_linear.set_defaults(func=linear)
    parser_linear.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_linear.add_argument(
        "-j", "--num_workers", type=int, required=False, default=4
    )
    parser_linear.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_linear.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    # post training quant
    parser_ptq = subparsers.add_parser(
        "ptq", help="the entrance of Post Training Quant"
    )
    parser_ptq.set_defaults(func=ptq)
    parser_ptq.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_ptq.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )
    parser_ptq.add_argument("-j", "--num_workers", type=int, required=False, default=4)


    # quantization aware SSL training
    parser_ssl = subparsers.add_parser(
        "ssl", help="the entrance of SSL Training"
    )
    parser_ssl.set_defaults(func=ssl)
    parser_ssl.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_ssl.add_argument("-j", "--num_workers", type=int, required=False, default=4)
    parser_ssl.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_ssl.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    # quantization aware SSL training - new version
    parser_draw = subparsers.add_parser(
        "draw", help="the entrance of SSL Training"
    )
    parser_draw.set_defaults(func=draw)
    parser_draw.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_draw.add_argument("-j", "--num_workers", type=int, required=False, default=4)
    parser_draw.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_draw.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    args = parser.parse_args()
    args.func(args)