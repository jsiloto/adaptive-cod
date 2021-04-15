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
from models.efficientdet.utils.utils import init_weights

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
    argparser.add_argument('-require_weights', action='store_true', help='Require preexisting weigths (used for eval)')
    argparser.add_argument('-ignore_optimizer', action='store_true', help='')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def freeze_modules(student_model, student_model_config, reset_unfrozen=False):
    if 'frozen_modules' in student_model_config:
        for student_path in student_model_config['frozen_modules']:
            student_module = module_util.get_module(student_model, student_path)
            module_util.freeze_module_params(student_module)

    elif 'unfrozen_modules' in student_model_config:
        module_util.freeze_module_params(student_model)
        for student_path in student_model_config['unfrozen_modules']:
            student_module = module_util.get_module(student_model, student_path)
            module_util.unfreeze_module_params(student_module)
            if reset_unfrozen:
                print("Reinitializing module: {}".format(student_path))
                init_weights(student_module)

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
    use_bottleneck_transformer = args.transform_bottleneck
    best_val_map = 0.0
    if file_util.check_if_exists(ckpt_file_path):
        if args.ignore_optimizer:
            best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=None, lr_scheduler=None)
        else:
            best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    num_epochs = train_config['num_epochs']
    log_freq = train_config['log_freq']
    teacher_model_without_dp = teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model
    student_model_without_ddp = \
        student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
    start_time = time.time()

    post_bn = False
    if 'post_batch_norm' in config['train']:
        post_bn = config['train']['post_batch_norm']

    for epoch in range(lr_scheduler.last_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        teacher_model.eval()
        student_model.train()
        teacher_model_without_dp.distill_backbone_only = distill_backbone_only
        student_model_without_ddp.distill_backbone_only = distill_backbone_only
        set_bottleneck_transformer(student_model_without_ddp, False)

        distill_model(distillation_box, train_data_loader, optimizer, log_freq, device, epoch)
        student_model_without_ddp.distill_backbone_only = False
        set_bottleneck_transformer(student_model_without_ddp, use_bottleneck_transformer)

        val_map = 0
        width_list = [1.0]
        if 'slimmable' in student_config['backbone']['params']:
            width_list = [0.25, 0.5, 0.75, 1.0]
            width_list = [w for w in width_list if w in student_config['backbone']['params']['width_mult_list']]

        for width in width_list:
            set_width(student_model, width)
            if post_bn:
                ComputeBN(student_model, train_data_loader)
            print('\n[Student model@width={}]'.format(width))
            coco_evaluator = main_util.evaluate(student_model, val_data_loader, device=device)
            val_map += coco_evaluator.coco_eval['bbox'].stats[0]
        val_map = val_map / len(width_list)

        print('BBox mAP: {:.4f})'.format(val_map))
        if val_map > best_val_map and misc_util.is_main_process():
            print('Updating ckpt (Best BBox mAP: {:.4f} -> {:.4f})'.format(best_val_map, val_map))
            best_val_map = val_map
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler, best_val_map, config, args,
                      ckpt_file_path)

        lr_scheduler.step()

    if distributed:
        dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def evaluate(teacher_model, student_model, test_data_loader, train_data_loader,
             device, student_only, use_bottleneck_transformer, student_model_config, post_bn=False):
    teacher_model_without_dp = teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model
    student_model_without_ddp = \
        student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
    teacher_model_without_dp.distill_backbone_only = False
    student_model_without_ddp.distill_backbone_only = False
    set_bottleneck_transformer(student_model_without_ddp, use_bottleneck_transformer)

    if not student_only:
        print('[Teacher model]')
        main_util.evaluate(teacher_model, test_data_loader, device=device)

    if "slimmable" in student_model_config['backbone']['params']:
        width_mult_list = student_model_config['backbone']['params']['width_mult_list']
        for width in width_mult_list:
            print('\n[Student model@width={}]'.format(width))
            set_width(student_model, width)
            if post_bn:
                ComputeBN(student_model, train_data_loader)
            main_util.evaluate(student_model, test_data_loader, device=device)

    else:
        main_util.evaluate(student_model, test_data_loader, device=device)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    teacher_model = get_model(config['teacher_model'], device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = get_model(student_model_config, device, require_weights=args.require_weights)

    reset_unfrozen = False
    if 'reset_unfrozen' in student_model_config:
        reset_unfrozen = student_model_config['reset_unfrozen']
    freeze_modules(student_model, student_model_config, reset_unfrozen=reset_unfrozen)
    set_bottleneck_transformer(student_model, True)
    post_bn = False
    if 'post_batch_norm' in config['train']:
        post_bn = config['train']['post_batch_norm']

    # print('Updatable parameters: {}'.format(module_util.get_updatable_param_names(student_model)))
    distill_backbone_only = student_model_config['distill_backbone_only']
    train_config = config['train']

    train_sampler, train_data_loader, val_data_loader, test_data_loader = \
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed)
    if distributed:
        teacher_model = DataParallel(teacher_model, device_ids=device_ids)
        student_model = DistributedDataParallel(student_model, device_ids=device_ids)

    if args.distill:
        distill(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader,
                device, distributed, distill_backbone_only, config, args)
        load_ckpt(config['student_model']['ckpt'],
                  model=student_model.module if isinstance(student_model, DistributedDataParallel) else student_model)
    evaluate(teacher_model, student_model, test_data_loader, train_data_loader, device,
             args.skip_teacher_eval, args.transform_bottleneck, student_model_config, post_bn)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
