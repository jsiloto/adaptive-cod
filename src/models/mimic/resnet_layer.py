from torch import nn

from models.slimmable.slimmable_ops import USConv2d, USBatchNorm2d
from models.ext.classifier import Ext4ResNet
from models.mimic.base import BottleneckBase4Ext, ExtEncoder, BottleneckIdentity

class Bottleneck4SmallResNet(BottleneckBase4Ext):
    def __init__(self, bottleneck_channel, ext_config, bottleneck_transformer):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class Bottleneck4LargeResNet(BottleneckBase4Ext):
    def __init__(self, bottleneck_channel, ext_config, bottleneck_transformer):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()

class SlimmableBottleneck4LargeResNet(BottleneckBase4Ext):
    def __init__(self, width_mult_list, ext_config, bottleneck_transformer):
        bottleneck_channel = 12

        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            USConv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False, slimmable_input=False)
        )
        decoder = nn.Sequential(
            USBatchNorm2d(bottleneck_channel, width_mult_list),
            nn.ReLU(inplace=True),
            USConv2d(bottleneck_channel, 64, kernel_size=2, bias=False, slimmable_output=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class FullySlimmableBottleneck4LargeResNet(BottleneckBase4Ext):
    def __init__(self, width_mult_list, ext_config, bottleneck_transformer):
        bottleneck_channel = 12

        encoder = nn.Sequential(
            USConv2d(64, 64, kernel_size=2, padding=1, bias=False),
            USBatchNorm2d(64, width_mult_list),
            USConv2d(64, 256, kernel_size=2, padding=1, bias=False),
            USBatchNorm2d(256, width_mult_list),
            nn.ReLU(inplace=True),
            USConv2d(256, 64, kernel_size=2, padding=1, bias=False),
            USBatchNorm2d(64, width_mult_list),
            USConv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            USBatchNorm2d(bottleneck_channel, width_mult_list),
            nn.ReLU(inplace=True),
            USConv2d(bottleneck_channel, 64, kernel_size=2, bias=False, slimmable_output=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class FullySlimmableLayer0(nn.Module):

    def __init__(self, width_mult_list):
        super().__init__()
        self.inplanes = 64
        self.conv1 = USConv2d(3, self.inplanes, kernel_size=7, stride=2,
                              padding=3, bias=False, slimmable_input=False)
        self.bn1 = USBatchNorm2d(self.inplanes, width_mult_list)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


def get_mimic_layers(backbone_name, backbone_config, bottleneck_transformer=None):
    layer0, layer1, layer2, layer3, layer4 = None, None, None, None, None
    backbone_params_config = backbone_config['params']
    layer0_config = backbone_params_config.get('layer0', None)
    layer1_config = backbone_params_config.get('layer1', None)

    if layer0_config is not None:
        layer0_name = layer0_config['name']
        if layer0_name == 'FullySlimmableLayer0':
            layer0 = FullySlimmableLayer0(layer0_config['width_mult_list'])
        else:
            raise ValueError('layer1_name `{}` is not expected'.format(layer0_name))

    if layer1_config is not None:
        layer1_name = layer1_config['name']
        ext_config = backbone_config.get('ext_config', None)
        if layer1_name == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], ext_config, bottleneck_transformer)
        elif layer1_name == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], ext_config, bottleneck_transformer)
        elif layer1_name == 'SlimmableBottleneck4LargeResNet' \
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = SlimmableBottleneck4LargeResNet(layer1_config['width_mult_list'], ext_config, bottleneck_transformer)
        elif layer1_name == 'FullySlimmableBottleneck4LargeResNet' \
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = FullySlimmableBottleneck4LargeResNet(layer1_config['width_mult_list'], ext_config, bottleneck_transformer)
        elif layer1_name == 'BottleneckIdentity' \
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = BottleneckIdentity(backbone_config, bottleneck_transformer)

        else:
            raise ValueError('layer1_name `{}` is not expected'.format(layer1_name))
    return layer0, layer1, layer2, layer3, layer4
