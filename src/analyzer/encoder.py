import torch
from torch import nn

class FasterRCNNEncoder(nn.Module):
    def __init__(self, student_model):
        super(FasterRCNNEncoder, self).__init__()
        student_model = student_model.backbone.body
        self.layer0 = student_model.layer0
        self.encoder = student_model.layer1.encoder
        self.output_shape = None
        self.original_output_shape = None
        self.use_encoder = True

    def forward(self, images, targets=None):
        x = images
        x = self.layer0(x)
        self.original_output_shape = x.shape
        if self.use_encoder:
            x = self.encoder(x)
        self.output_shape = x.shape
        return x

class EfficientDetEncoder(nn.Module):
    def __init__(self, student_model):
        super(EfficientDetEncoder, self).__init__()
        effnet = student_model.backbone_net
        self._conv_stem = effnet.model._conv_stem
        self._bn0 = effnet.model._bn0
        self.encoder = effnet.bottleneck.encoder

        self.original_output_shape = None
        self.output_shape = None
        self.use_encoder = True

        self._blocks = []
        num_bottlenecks = 0
        for idx, block in enumerate(effnet.model._blocks):
            if block._depthwise_conv.stride == [2, 2]:
                num_bottlenecks += 1
                if num_bottlenecks == 3:
                    break
            self._blocks.append(block)
        self._blocks = torch.nn.Sequential(*self._blocks)

    def forward(self, images, targets=None):
        x = images
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._blocks(x)
        self.original_output_shape = x.shape
        if self.use_encoder:
            x = self.encoder(x)
        self.output_shape = x.shape
        return x

def full_encoder(student_model, student_model_config):
    encoders = {
        "efficientdet": EfficientDetEncoder,
        "faster_rcnn": FasterRCNNEncoder,
        "mask_rcnn": FasterRCNNEncoder,
    }
    return encoders[student_model_config['name']](student_model)
