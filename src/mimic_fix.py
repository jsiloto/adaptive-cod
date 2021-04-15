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


def freeze_batch_norm_outside_bottleneck(model):
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
    model.backbone_net.bottleneck.train()


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('-distill', action='store_true', help='distill a teacher model')
    argparser.add_argument('-skip_teacher_eval', action='store_true', help='skip teacher model evaluation in testing')
    argparser.add_argument('-transform_bottleneck', action='store_true',
                           help='use bottleneck transformer (if defined in yaml) in testing')
    argparser.add_argument('-post_bn', action='store_true', help='use post traing batch norm calculation')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def freeze_modules(student_model, student_model_config):
    if 'frozen_modules' in student_model_config:
        for student_path in student_model_config['frozen_modules']:
            student_module = module_util.get_module(student_model, student_path)
            module_util.freeze_module_params(student_module)

    elif 'unfrozen_modules' in student_model_config:
        module_util.freeze_module_params(student_model)
        for student_path in student_model_config['unfrozen_modules']:
            student_module = module_util.get_module(student_model, student_path)
            module_util.unfreeze_module_params(student_module)


def distill_model(distillation_box, data_loader, optimizer, log_freq, device, epoch):
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000.0
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = main_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, log_freq, header):
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


def distill(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader,
            device, distributed, distill_backbone_only, config, args):
    train_config = config['train']
    student_config = config['student_model']
    distillation_box = DistillationBox(teacher_model, student_model,
                                       train_config['criterion'], student_config)
    ckpt_file_path = config['student_model']['ckpt']
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
        save_ckpt(student_model, optimizer, lr_scheduler, best_val_map, config, args, ckpt_file_path)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    teacher_model = get_model(config['teacher_model'], device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = get_model(student_model_config, device)
    freeze_modules(student_model, student_model_config)
    ckpt_file_path = config['student_model']['ckpt']
    train_config = config['train']
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
        save_ckpt(student_model, optimizer, lr_scheduler, best_val_map, config, args, ckpt_file_path)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
