import random

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

import random
from distillation.loss import get_loss
from models.org.rcnn import KeypointRCNN
from myutils.pytorch import module_util
from models.mimic.base import set_width


class DistillationBox(nn.Module):
    def __init__(self, teacher_model, student_model, criterion_config, student_config):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.target_module_pairs = list()
        self.slimmable = ('slimmable' in student_config['backbone']['params'])
        self.student_config = student_config

        def extract_output(self, input, output):
            self.__dict__['distillation_box']['output'] = output

        teacher_model_without_dp = teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model
        student_model_without_ddp = \
            student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
        for loss_name, loss_config in criterion_config['terms'].items():
            teacher_path, student_path = loss_config['ts_modules']
            self.target_module_pairs.append((teacher_path, student_path))
            teacher_module = module_util.get_module(teacher_model_without_dp, teacher_path)
            student_module = module_util.get_module(student_model_without_ddp, student_path)
            teacher_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': teacher_path,
                                                           'is_teacher': True}
            student_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': student_path,
                                                           'is_teacher': False}
            teacher_module.register_forward_hook(extract_output)
            student_module.register_forward_hook(extract_output)

        self.criterion = get_loss(criterion_config)
        self.require_adjustment = isinstance(student_model_without_ddp, KeypointRCNN)


    def forward(self, images, targets):
        teacher_model_without_dp = \
            self.teacher_model.module if isinstance(self.teacher_model, DataParallel) else self.teacher_model
        student_model_without_ddp = \
            self.student_model.module if isinstance(self.student_model, DistributedDataParallel) else self.student_model

        self.teacher_model(images)
        total_loss = 0

        train_widths = [1.0]
        if self.slimmable:
            width_mult_list = self.student_config['backbone']['params']['width_mult_list']
            width_copies = self.student_config['backbone']['params']['width_copies']
            width_mult_list.sort()
            train_widths = [width_mult_list[0]]
            for i in range(width_copies-2):
                train_widths.append(random.choice(width_mult_list))
            train_widths.append(width_mult_list[-1])

        for width_mult in train_widths:
            set_width(self.student_model, width_mult)
            org_loss_dict = self.student_model(images, targets)
            output_dict = dict()
            for teacher_path, student_path in self.target_module_pairs:
                partial_teacher_module = module_util.get_module(teacher_model_without_dp, teacher_path)
                partial_student_module = module_util.get_module(student_model_without_ddp, student_path)
                teacher_dict = partial_teacher_module.__dict__['distillation_box']
                student_dict = partial_student_module.__dict__['distillation_box']
                output_dict[teacher_dict['loss_name']] = ((teacher_dict['path_from_root'], teacher_dict['output']),
                                                          (student_dict['path_from_root'], student_dict['output']))

            total_loss += self.criterion(output_dict, org_loss_dict)

        # Free up memory
        for teacher_path, student_path in self.target_module_pairs:
            partial_teacher_module = module_util.get_module(teacher_model_without_dp, teacher_path)
            partial_student_module = module_util.get_module(student_model_without_ddp, student_path)
            teacher_dict = partial_teacher_module.__dict__['distillation_box']
            student_dict = partial_student_module.__dict__['distillation_box']
            partial_teacher_module.__dict__['distillation_box']['output'] = None
            partial_student_module.__dict__['distillation_box']['output'] = None

        return total_loss