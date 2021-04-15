# Author: Zylo117

import numpy as np
import torch
from torch import nn

from models.efficientdet.efficientnet.utils import efficientnet_params, efficientnet
from models.efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from models.efficientdet.model_utils import Anchors, BBoxTransform, ClipBoxes
from models.efficientdet.utils.utils import preprocess, postprocess, invert_affine
from models.efficientdet.loss import FocalLoss
from models.mimic.base import BottleneckIdentity
from models.mimic.efficientdet_layer import Bottlenet4EfficientDet, \
    SlimmableBottlenet4EfficientDet, MBConvBlockEfficientDet, MBConvBlockDecoderOnlyEfficientDet, \
    Bottleneck4EfficientDet, ChannelSelect
from models.slimmable.slimmable_ops import USMBConvBlock


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.criterion = FocalLoss()
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.input_size = self.input_sizes[self.compound_coef]
        self.distill_backbone_only = False
        self.fully_slimmable = False

        self.p3 = torch.nn.Identity()
        self.p4 = torch.nn.Identity()
        self.p5 = torch.nn.Identity()

        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def forward(self, images, targets=None):
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        if isinstance(images, list):
            device = images[0].device
        else:
            device = images.device
        ori_imgs, framed_imgs, framed_metas = preprocess(images, max_size=input_sizes[self.compound_coef])

        framed_imgs = torch.from_numpy(np.array(framed_imgs))
        framed_imgs = framed_imgs.permute(0, 3, 1, 2)
        images = framed_imgs.to(device)
        p3, p4, p5 = self.backbone_net(images)

        p3 = self.p3(p3)
        p4 = self.p4(p4)
        p5 = self.p5(p5)

        features = (p3, p4, p5)
        if self.distill_backbone_only:
            return features

        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(images, images.dtype)

        if self.training:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, targets)
            loss_dict = {
                'cls_loss': cls_loss,
                'reg_loss': reg_loss,
            }
            return loss_dict

        else:
            preds = postprocess(framed_imgs, anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold=0.05, iou_threshold=0.5)
            if not preds:
                return {}
            preds = invert_affine(framed_metas, preds)
            detection = []
            for p in preds:
                d = {
                    'boxes': torch.tensor(p['rois']),
                    'labels': torch.tensor(p['class_ids']) + 1,
                    'scores': torch.tensor(p['scores'])
                }
                detection.append(d)
            return detection

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

    def load_and_patch_state_dict(self, state_dict, strict=True):
        if self.fully_slimmable:
            for idx, block in enumerate(self.backbone_net.model._blocks):
                if isinstance(block, USMBConvBlock) and block.fully_slimmable:
                    state_dict = self.patch_state_dict_block(state_dict, idx, block)
                    pass
        super().load_state_dict(state_dict, strict)

    def patch_state_dict_block(self, state_dict, idx, block):
        def patch_bn(bn_idx):
            prefix = 'backbone_net.model._blocks.{}._bn{}.'.format(idx, bn_idx)
            for i in range(len(block.width_mult_list)):
                num_features = len(state_dict[prefix + 'weight'])
                size = int(round(num_features * block.width_mult_list[i]))
                # state_dict[prefix + 'bn.{}.'.format(i) + 'weight'] = state_dict[prefix + 'weight'][:size]
                # state_dict[prefix + 'bn.{}.'.format(i) + 'bias'] = state_dict[prefix + 'bias'][:size]
                state_dict[prefix + 'bn.{}.'.format(i) + 'running_mean'] = state_dict[prefix + 'running_mean'][:size]
                state_dict[prefix + 'bn.{}.'.format(i) + 'running_var'] = state_dict[prefix + 'running_var'][:size]
                state_dict[prefix + 'bn.{}.'.format(i) + 'num_batches_tracked'] = state_dict[
                    prefix + 'num_batches_tracked']

        patch_bn(0)
        patch_bn(1)
        patch_bn(2)
        return state_dict


def get_mimic_layer(config, bottleneck_transformer):
    constructor = {
        "Bottlenet4EfficientDet": Bottlenet4EfficientDet,
        "BottleneckIdentity": BottleneckIdentity,
        "SlimmableBottlenet4EfficientDet": SlimmableBottlenet4EfficientDet,
        "MBConvBlockEfficientDet": MBConvBlockEfficientDet,
        "MBConvBlockDecoderOnlyEfficientDet": MBConvBlockDecoderOnlyEfficientDet,
        "Bottleneck4EfficientDet": Bottleneck4EfficientDet,
        "ChannelSelect": ChannelSelect,
    }
    return constructor[config['bottleneck']['name']](config, bottleneck_transformer)


def patch_slimmable_layers(model, config):
    global_params = model.backbone_net.model._global_params
    new_blocks = nn.ModuleList([])
    num_bottlenecks = 0
    slimmable_input = False
    for idx, block in enumerate(model.backbone_net.model._blocks):
        if block._depthwise_conv.stride == [2, 2]:
            num_bottlenecks += 1

        if num_bottlenecks == 0 or num_bottlenecks >= 3:
            new_blocks.append(block)
        else:
            new_block = USMBConvBlock(block._block_args, global_params,
                                      config['width_mult_list'],
                                      fully_slimmable=True,
                                      slimmable_input=slimmable_input,
                                      slimmable_output=True)
            new_blocks.append(new_block)
            slimmable_input = True

    model.backbone_net.model._blocks = new_blocks
    model.fully_slimmable = True
    return model


def get_model(*args, **kwargs):
    backbone_params_config = kwargs['backbone_config']['params']
    bottleneck_transformer = kwargs['bottleneck_transformer']
    compound_coef = backbone_params_config['compound_coef']

    anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                                 ratios=anchors_ratios, scales=anchors_scales)

    if 'drop_connect' in backbone_params_config:
        model.backbone_net.model._global_params = \
            model.backbone_net.model._global_params._replace(
                drop_connect_rate=backbone_params_config['drop_connect'])

    if 'fully_slimmable' in backbone_params_config:
        print("Patching Slimmable Layers")
        model = patch_slimmable_layers(model, backbone_params_config)

    if backbone_params_config['pretrained']:
        if model.fully_slimmable:
            model.load_and_patch_state_dict(torch.load(backbone_params_config['weights']))
        else:
            model.load_state_dict(torch.load(backbone_params_config['weights']))

    if 'bottleneck' in backbone_params_config:
        bottleneck = get_mimic_layer(backbone_params_config, bottleneck_transformer)
        model.backbone_net.bottleneck = bottleneck

    return model
