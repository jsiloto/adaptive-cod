import argparse
import datetime
import time
import os
import sys
import numpy as np
import torch
from torch import nn
from models import load_ckpt, get_model, save_ckpt
from myutils.common import file_util, yaml_util
from utils import data_util, main_util, misc_util
from models.slimmable.slimmable_ops import USConv2d, USBatchNorm2d, USConv2dStaticSamePadding
from models.mimic.base import set_width
from ptflops import get_model_complexity_info
from myutils.pytorch import func_util, module_util
from analyzer.hooks import usconv_flops_counter_hook, bn_flops_counter_hook
from analyzer.encoder import full_encoder
from analyzer.analysis import hide_prints
import warnings
from analyzer.analysis import model_analysis
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path

def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--config', required=False, help='yaml file path')
    argparser.add_argument('--dir', required=False, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-debug', action='store_true', help='')
    return argparser


def summarize(results):
    size = results['input_size']
    jpeg_size = results["jpeg_size"]

    macs_encoder = results['macs_base_encoder']+  results['macs_compressor']
    params_encoder = results['params_base_encoder'] + results['params_compressor']

    macs_full = results["macs_decoder"] + macs_encoder
    params_full = results["params_decoder"] + params_encoder

    print("{}: input shape ({} Bytes)".format((1, 3, size, size), size * size * 3))
    print("{}: output shape".format(results['output_shape']))
    print("{} Encoder Bytes {:.2f}%: Compression".format(np.prod(results["output_shape"]), results["compression"]  * 100))
    print("{} JPEG Bytes {:.2f}%: JPEG 95 Compression".format(jpeg_size, results["jpeg_compression"]  * 100))
    print("{:<30}  {:.1f} GMacs {:.1f}k params".format("Base Encoder:", results['macs_base_encoder'] / 1e9, results['params_base_encoder']/ 1e3))
    print("{:<30}  {:.1f} GMacs {:.1f}k params".format("Compression Encoder:", results['macs_compressor']  / 1e9, results['params_compressor'] / 1e3))
    print('{:<30}  {:.1f} GMacs {:.1f}k params'.format('Full Encoder: ',macs_encoder/1e9, params_encoder/ 1e3))
    print('{:<30}  {:.1f} GMacs {:.1f}k params'.format('Decoder: ',  results["macs_decoder"]/ 1e9,results["params_decoder"] / 1e3))
    print('{:<30}  {:.1f} GMacs {:.1f}k params'.format('Full Model: ', macs_full / 1e9, params_full / 1e3))



def main(args):
    assert not (args.config and args.dir)

    if args.config:
        print(args.config)
        config = yaml_util.load_yaml_file(args.config)

        width_list = [1.0]
        if 'slimmable' in config['student_model']['backbone']['params']:
            width_list = config['student_model']['backbone']['params']['width_mult_list']


        results = model_analysis(config, args.device, setting='Teacher', debug=False)
        print("************** Teacher *************")
        summarize(results)
        print("************** Student *************")
        for width in width_list:
            results = model_analysis(config, args.device, setting=width, debug=False)
            summarize(results)

    elif "dir" in args:
        result_dir = join(args.dir, "results")
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        config_file_list = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]
        config_file_list = [f for f in config_file_list if f.endswith('.yaml')]
        for config_file in config_file_list:
            result_file = join(result_dir, config_file).split('.yaml')[0] + ".csv"
            config_file = join(args.dir, config_file)

            print("#################################################################################")
            print("config_file: {},".format(config_file))
            print("result_file: {},".format(result_file))

            # Build Model
            config = yaml_util.load_yaml_file(config_file)
            student_model_config = config['student_model']
            # student_model = get_model(student_model_config, device, strict=False)
            # encoder = full_encoder(student_model, student_model_config)

            if isfile(result_file):
                df = pd.read_csv(result_file)
                df = df.loc[:, ~df.columns.str.match("Unnamed")]
                df.reset_index()
                df.set_index("Setting")
            else:
                continue

            new_df = []
            for index, row in df.iterrows():
                setting = row['Setting']
                results = model_analysis(config, args.device, setting=setting, debug=args.debug)

                # attempts = 0
                # while attempts < 3:
                #     try:
                #         results = model_analysis(config, device, setting=setting, debug=False)
                #         break
                #     except RuntimeError as e:
                #         attempts += 1
                #         print("{}".format(e.args))


                results.update(row.to_dict())
                new_df.append(results)
            new_df = pd.DataFrame(new_df)
            new_df.reset_index()
            new_df.set_index("Setting")
            new_df.to_csv(result_file)




if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
