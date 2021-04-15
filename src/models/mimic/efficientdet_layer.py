from torch import nn

from models.slimmable.slimmable_ops import USConv2d, USBatchNorm2d, USMBConvBlock, USChannelDrop, USChannelRestore
from models.ext.classifier import Ext4ResNet
from models.mimic.base import BottleneckBase4Ext, ExtEncoder
from models.efficientdet.efficientnet.model import MBConvBlock
from models.efficientdet.efficientnet.utils import BlockArgs, efficientnet_params, efficientnet, round_filters, \
    round_repeats, Swish

activation_dict = {
    "ReLU": nn.ReLU(inplace=True),
    "Swish": Swish()
}

def get_channel_number(compound_coef):
    channels = [40, 40, 48, 48, 56, 64, 72, 72, 80]
    return channels[compound_coef]

class ChannelSelect(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        strategy = config['bottleneck']['strategy']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)
        encoder = USChannelDrop(bottleneck_channel, strategy=strategy)
        decoder = USChannelRestore(original_channels, strategy=strategy)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class Bottlenet4EfficientDet(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)
        activation = config['bottleneck']['activation']
        activation_fn = activation_dict[activation]

        encoder = nn.Sequential(
            nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(original_channels),
            activation_fn,
            nn.Conv2d(original_channels, bottleneck_channel, kernel_size=1, bias=True),
            nn.BatchNorm2d(bottleneck_channel),
            activation_fn,
        )
        decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channel, original_channels, kernel_size=1, padding=1, bias=True),
            nn.BatchNorm2d(original_channels),
            activation_fn,
            nn.Conv2d(original_channels, original_channels, kernel_size=3, bias=True),
            nn.BatchNorm2d(original_channels),
        )
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class SlimmableBottlenet4EfficientDet(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)
        width_mult_list = config['width_mult_list']
        activation = config['bottleneck']['activation']
        activation_fn = activation_dict[activation]

        encoder = nn.Sequential(
            nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(original_channels),
            activation_fn,
            USConv2d(original_channels, bottleneck_channel, kernel_size=1, bias=True, slimmable_input=False),
            USBatchNorm2d(bottleneck_channel, width_mult_list),
            activation_fn
        )
        decoder = nn.Sequential(
            USConv2d(bottleneck_channel, original_channels, kernel_size=1, padding=1, bias=True,
                     slimmable_output=False),
            nn.BatchNorm2d(bottleneck_channel),
            activation_fn,
            nn.Conv2d(original_channels, original_channels, kernel_size=3, bias=True),
            nn.BatchNorm2d(original_channels),
        )
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()





class MBConvBlockDecoderOnlyEfficientDet(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)

        w, d, s, p = efficientnet_params("efficientnet-b{}".format(compound_coef))
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d,
                                                  dropout_rate=p, image_size=s)
        block_args = blocks_args[4]
        block_args_decoder = block_args._replace(
            input_filters=bottleneck_channel,
            output_filters=original_channels,
            num_repeat=round_repeats(block_args.num_repeat, global_params)
        )

        encoder = USChannelDrop(bottleneck_channel)
        decoder = MBConvBlock(block_args_decoder, global_params)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class MBConvBlockEfficientDet(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)
        slimmable = ('slimmable' in config)

        w, d, s, p = efficientnet_params("efficientnet-b{}".format(compound_coef))
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d,
                                                  dropout_rate=p, image_size=s)
        block_args = blocks_args[4]
        block_args_encoder = block_args._replace(
            input_filters=original_channels,
            output_filters=bottleneck_channel,
            num_repeat=round_repeats(block_args.num_repeat, global_params)
        )
        block_args_decoder = block_args._replace(
            input_filters=bottleneck_channel,
            output_filters=original_channels,
            num_repeat=round_repeats(block_args.num_repeat, global_params)
        )

        if slimmable:
            fully_slimmable = False
            if 'fully_slimmable' in config:
                fully_slimmable = config['fully_slimmable']
            width_mult_list = config['width_mult_list']
            print("fully_slimmable", fully_slimmable)
            encoder = USMBConvBlock(block_args_encoder, global_params, width_mult_list,
                                    slimmable_input=fully_slimmable, fully_slimmable=fully_slimmable)
            decoder = USMBConvBlock(block_args_decoder, global_params, width_mult_list,
                                    slimmable_output=False, fully_slimmable=fully_slimmable)
        else:
            encoder = MBConvBlock(block_args_encoder, global_params)
            decoder = MBConvBlock(block_args_decoder, global_params)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class Bottleneck4EfficientDet(BottleneckBase4Ext):
    def __init__(self, config, bottleneck_transformer=None):
        bottleneck_channel = config['bottleneck']['bottleneck_channel']
        compound_coef = config['compound_coef']
        original_channels = get_channel_number(compound_coef)
        activation = config['bottleneck']['activation']
        activation_fn = activation_dict[activation]

        encoder = nn.Sequential(
            nn.Conv2d(original_channels, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            activation_fn,
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            activation_fn,
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            activation_fn,
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, original_channels, kernel_size=2, bias=False),
            nn.BatchNorm2d(original_channels),
        )
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()

# class SlimmableMBConvBlockEfficientDet(BottleneckBase4Ext):
#     # Todo(jsiloto) refactor with above
#     def __init__(self, config, bottleneck_transformer=None):
#         bottleneck_channel = config['bottleneck']['bottleneck_channel']
#         compound_coef = config['compound_coef']
#         original_channels = get_channel_number(compound_coef)
#         width_mult_list = config['width_mult_list']
#
#         w, d, s, p = efficientnet_params("efficientnet-b{}".format(compound_coef))
#         blocks_args, global_params = efficientnet(
#             width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
#
#         for b in blocks_args:
#             print(b)
#
#         print(global_params)
#         exit()
#         block_args = blocks_args[4]
#         block_args_encoder = block_args._replace(
#             input_filters=original_channels,
#             output_filters=bottleneck_channel,
#             num_repeat=round_repeats(block_args.num_repeat, global_params)
#         )
#         block_args_decoder = block_args._replace(
#             input_filters=bottleneck_channel,
#             output_filters=original_channels,
#             num_repeat=round_repeats(block_args.num_repeat, global_params)
#         )
#
#         encoder = USMBConvBlock(block_args_encoder, global_params, width_mult_list, slimmable_input=False)
#         decoder = USMBConvBlock(block_args_decoder, global_params, width_mult_list, slimmable_output=False)
#         super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)
#
#     def get_ext_classifier(self):
#         return self.encoder.get_ext_classifier()
