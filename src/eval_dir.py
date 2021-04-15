import argparse
import datetime
import time

import torch
from torch import distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from distillation.tool import DistillationBox
from models import load_ckpt, get_model, save_ckpt, set_bottleneck_transformer
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from utils import data_util, main_util, misc_util
from models.mimic.base import set_width
from models.slimmable.compute_post_bn import ComputeBN
from torch.nn.modules.batchnorm import _BatchNorm

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path

def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--dir', required=True, help='yaml dir path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-resume', action='store_true', help='Do not Re-run experiments')
    argparser.add_argument('-dry_run', action='store_true', help='Validate configs without eval')
    return argparser

def distill_model(distillation_box, data_loader, optimizer, log_freq, device, epoch):
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000.0
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = main_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss = distillation_box(images, targets)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        # torch.cuda.empty_cache()


def evaluate(model, test_data_loader, device):
    coco_evaluator = main_util.evaluate(model, test_data_loader, device=device)
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        print("***************************************")
        print(coco_eval.stats)
        print(iou_type)

    return coco_evaluator.coco_eval['bbox'].stats

def experiment(config, device, args):
    # Load Model
    teacher_model = get_model(config['teacher_model'], device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = get_model(student_model_config, device, strict=True, require_weights=True)
    set_bottleneck_transformer(student_model, True)
    if args.dry_run:
        return None

    #Load Dataset
    train_config = config['train']
    train_sampler, train_data_loader, val_data_loader, test_data_loader = \
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed=False)



    # Prepare dataframe
    header = ["Setting", "Iou50:95", "Iou50"]
    df = pd.DataFrame(columns=header)
    res = evaluate(teacher_model, test_data_loader, device)
    df = df.append({"Setting": "Teacher",
                    "Iou50:95": res[0], "Iou50": res[1]}, ignore_index=True)

    post_bn = False
    if 'post_batch_norm' in config['train']:
        post_bn = config['train']['post_batch_norm']

    width_mult_list = [1.0]
    if "slimmable" in student_model_config['backbone']['params']:
        width_mult_list = student_model_config['backbone']['params']['width_mult_list']
    for width in width_mult_list:
        print('\n[Student model@width={}]'.format(width))
        set_width(student_model, width)
        if post_bn:
            ComputeBN(student_model, train_data_loader)
        res = evaluate(student_model, test_data_loader, device)
        df = df.append({"Setting": str(width),
                        "Iou50:95": res[0], "Iou50": res[1]}, ignore_index=True)

    return df

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    result_dir = join(args.dir, "results")
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    config_file_list = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]
    config_file_list = [f for f in config_file_list if f.endswith('.yaml')]
    config_file_list.sort()
    for config_file in config_file_list:
        result_file = join(result_dir, config_file).split('.yaml')[0]+".csv"
        config_file = join(args.dir, config_file)

        print("#################################################################################")
        print("config_file: {},".format(config_file))
        print("result_file: {},".format(result_file))
        config = yaml_util.load_yaml_file(config_file)
        if args.resume and os.path.isfile(result_file):
            print("Experiment Done: Skipping")
            continue
        results = experiment(config, device, args)
        if args.dry_run:
            continue

        print(results)
        results.to_csv(result_file, float_format='%.4f')

if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
