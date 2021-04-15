import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch
from functools import partial
from models.efficientdet.efficientnet.model import MBConvBlock

from models.efficientdet.efficientnet.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    get_same_padding_conv2d,
    Swish,
    MemoryEfficientSwish,
)

"""
forked from Mutualnet Repo:
https://github.com/taoyang1122/MutualNet/tree/master/models/slimmable_ops.py
"""


def float_index(a, floats, **kwargs):
    l = np.isclose(a, floats, rtol=0.01, **kwargs).nonzero()[0]
    if len(l) == 0:
        return None
    if len(l) == 1:
        return l[0]
    else:
        raise ValueError


def make_divisible(v, divisor=3, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def close_to_any(a, floats, **kwargs):
    return np.any(np.isclose(a, floats, **kwargs))

class USChannelDrop(nn.Module):
    def __init__(self, num_channels, strategy='naive'):
        self.num_channels = num_channels
        self.width_mult=1.0
        self.strategy = strategy
        super().__init__()

    def forward(self, input):
        channels = int(round(self.num_channels * self.width_mult, 0))

        if self.strategy =='naive':
            y = input[:,:channels,:,:]

        elif self.strategy == 'magnitude':
            magnitude = torch.sum(torch.square(input), dim=(-1, -2))
            sorted = torch.sort(magnitude, dim=-1, descending=True)[0]
            threshold = sorted[:, channels-1]
            mask = magnitude > threshold
            y = input
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if not mask[i, j]:
                        y[i][j] = torch.zeros(y.shape[2:4])
        return y

class USChannelRestore(nn.Module):
    def __init__(self, output_channels, strategy='naive'):
        self.output_channels = output_channels
        self.width_mult=1.0
        self.strategy = strategy
        super().__init__()

    def forward(self, input):
        if self.strategy == 'naive':
            s = input.shape
            zeros = torch.zeros((s[0], self.output_channels-s[1], s[2], s[3]), device=input.device)
            y = torch.cat((input, zeros), dim=1)
        elif self.strategy == 'magnitude':
            y = input
        return y






class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1],
                 slimmable_input=True, slimmable_output=True):
        in_channels_max = in_channels
        out_channels_max = out_channels
        groups = in_channels_max if depthwise else 1
        super(USConv2d, self).__init__(
            in_channels_max, out_channels_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.width_mult = 1.0
        self.us = us
        self.ratio = ratio
        self.slimmable_input = slimmable_input
        self.slimmable_output = slimmable_output

    def forward(self, input):
        in_channels = self.in_channels_basic
        out_channels = self.out_channels_basic

        if self.slimmable_input:
            in_channels = int(round(self.in_channels_basic * self.width_mult, 0))
        if self.slimmable_output:
            out_channels = int(round(self.out_channels_basic * self.width_mult, 0))

        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]

        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias

        y = nn.functional.conv2d(input, weight, bias,
                                 self.stride, self.padding,
                                 self.dilation, self.groups)

        # if getattr(FLAGS, 'conv_averaged', False):
        #     y = y * (max(self.in_channels_list)/self.in_channels)
        return y


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_mult_list, ratio=1, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        num_features_max = int(round(num_features, 0))
        super(USBatchNorm2d, self).__init__(num_features_max, momentum=momentum, affine=affine, eps=eps,
                                            track_running_stats=track_running_stats)
        self.num_features_basic = num_features
        self.width_mult_list = width_mult_list
        # for tracking log during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [int(round(num_features * width_mult)) for width_mult in width_mult_list]
             ]
        )
        self.ratio = ratio
        self.width_mult = 1.0
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = int(round(self.num_features_basic * self.width_mult, 0))
        idx = float_index(self.width_mult, self.width_mult_list)

        if idx is not None:
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y


#####################################################################################

class USConv2dStaticSamePadding(nn.Module):
    # TODO(jsiloto) Figure out license for this
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, groups=1, dilation=1, depthwise=False,
                 slimmable_input=True, slimmable_output=True,
                 **kwargs):
        super().__init__()
        in_channels_max = in_channels
        out_channels_max = out_channels
        groups = in_channels_max if depthwise else groups
        self.conv = nn.Conv2d(in_channels_max, out_channels_max, kernel_size, stride=stride,
                              dilation=dilation, bias=bias, groups=groups)

        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.slimmable_input = slimmable_input
        self.slimmable_output = slimmable_output
        self.width_mult = 1.0
        self.depthwise = depthwise
        self.bias = bias
        self.groups = groups

        if self.depthwise:
            self.slimmable_input = True
            self.slimmable_output = True

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        in_channels = self.in_channels_basic
        out_channels = self.out_channels_basic

        if self.slimmable_input:
            in_channels = int(round(self.in_channels_basic * self.width_mult, 0))
        if self.slimmable_output:
            out_channels = int(round(self.out_channels_basic * self.width_mult, 0))

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        weight = self.conv.weight[:out_channels, :in_channels, :, :]

        if self.conv.bias is not None:
            bias = self.conv.bias[:out_channels]
        else:
            bias = self.conv.bias

        self.groups = in_channels if self.depthwise else 1
        y = nn.functional.conv2d(x, weight, bias,
                                 self.stride, self.conv.padding,
                                 self.dilation, self.groups)

        return y


class USMBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, width_mult_list,
                 fully_slimmable=False, slimmable_input=True, slimmable_output=True):
        super().__init__()
        self.fully_slimmable = fully_slimmable
        self._block_args = block_args
        self.width_mult_list = width_mult_list
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        USConv2d = partial(USConv2dStaticSamePadding, image_size=global_params.image_size)
        if fully_slimmable:
            Conv2d = USConv2d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if self._block_args.expand_ratio != 1:
            self._expand_conv = USConv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False,
                                         slimmable_input=slimmable_input, slimmable_output=fully_slimmable)
            if fully_slimmable:
                self._bn0 = USBatchNorm2d(num_features=oup, width_mult_list=width_mult_list,
                                          momentum=self._bn_mom, eps=self._bn_eps)
            else:
                self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup, depthwise=True,
            kernel_size=k, stride=s, bias=False)

        if fully_slimmable:
            self._bn1 = USBatchNorm2d(num_features=oup, width_mult_list=width_mult_list,
                                      momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn1 = nn.BatchNorm2d(num_features=oup,
                                       momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = USConv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False,
                                      slimmable_output=slimmable_output, slimmable_input=fully_slimmable)

        if slimmable_output:
            self._bn2 = USBatchNorm2d(num_features=final_oup, width_mult_list=width_mult_list,
                                      momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn2 = nn.BatchNorm2d(num_features=final_oup,
                                       momentum=self._bn_mom, eps=self._bn_eps)

        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                print("Drop Connect: Shouldnt happen")
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

    # def load_state_dict(self, state_dict, strict=False):
    #     print("HERE")
    #     exit()
    #
    # def set_weights(self, block):
    #     state_dict = block.state_dict()
    #     self.abc(state_dict)
    #     # self.load_state_dict(block.state_dict(), strict=False)
    #
    # def abc(self, state_dict):
    #     for i in range(len(self.width_mult_list)):
    #         num_features = len(state_dict['_bn0.weight'])
    #         size = int(round(num_features * self.width_mult_list[i]))
    #         state_dict["_bn0.bn.{}.weight".format(i)] = state_dict['_bn0.weight'][:size]
    #         state_dict["_bn0.bn.{}.bias".format(i)] = state_dict['_bn0.bias'][:size]
    #         state_dict["_bn0.bn.{}.running_mean".format(i)] = state_dict['_bn0.running_mean'][:size]
    #         state_dict["_bn0.bn.{}.running_var".format(i)] = state_dict['_bn0.running_var'][:size]
    #         state_dict["_bn0.bn.{}.num_batches_tracked".format(i)] = state_dict['_bn0.num_batches_tracked']
    #
