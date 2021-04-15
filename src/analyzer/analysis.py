from contextlib import nullcontext
import os
import sys
import numpy as np
import torch
from models import load_ckpt, get_model, save_ckpt
from models.slimmable.slimmable_ops import USConv2d, USBatchNorm2d, USConv2dStaticSamePadding
from models.mimic.base import set_width
from ptflops import get_model_complexity_info
from myutils.pytorch import func_util, module_util
from analyzer.hooks import usconv_flops_counter_hook, bn_flops_counter_hook
from analyzer.encoder import full_encoder
from copy import deepcopy

def hide_prints(debug=False):
    if debug:
        return nullcontext
    else:
        return HiddenPrints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_trainable_modules(student_model, student_model_config):
    if 'frozen_modules' in student_model_config:
        all_modules = list(student_model.modules())
        modules = []
        for student_path in student_model_config['frozen_modules']:
            m = module_util.get_module(student_model, student_path)
            modules.append(module_util.get_module(student_model, student_path))

        trainable_modules = [m for m in all_modules if m not in modules]
        return trainable_modules

    elif 'unfrozen_modules' in student_model_config:
        trainable_modules = []
        for student_path in student_model_config['unfrozen_modules']:
            trainable_modules.append(module_util.get_module(student_model, student_path))
        return trainable_modules

def jpeg_size_example(input_size):
    from io import BytesIO
    from PIL import Image
    im1 = Image.open("resource/stock_photo.jpg")
    im1 = im1.resize((input_size, input_size), Image.ANTIALIAS)
    buffer = BytesIO()
    im1.save(buffer, format="JPEG", quality=95)
    buffer.seek(0, os.SEEK_END)
    size = buffer.tell()
    return size


def model_analysis(config, device, setting=1.0, debug=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # BoilerPlate
    custom_module_mapping = {
        USConv2d: usconv_flops_counter_hook,
        USConv2dStaticSamePadding: usconv_flops_counter_hook,
        USBatchNorm2d: bn_flops_counter_hook,
    }

    student_model_config = config['student_model']
    if setting == 'Teacher':
        width = 1.0
    else:
        width = float(setting)

    num_bits = config['student_model']['bottleneck_transformer']['components']['quantizer']['params']['num_bits']

    # Build Model
    student_model = get_model(student_model_config, device, strict=False)
    encoder = full_encoder(student_model, student_model_config)

    # Analyze
    results = {}
    size = student_model.input_size
    jpeg_size = jpeg_size_example(size)
    print("Width = {}".format(width))
    set_width(student_model, width)

    def input_constructor(input_res):
        batch = torch.rand((1, *input_res),dtype=next(student_model.parameters()).dtype,
                                         device=device)
        return {"images": batch}



    with hide_prints(debug)():
        encoder.use_encoder = False
        macs_base_encoder, params_base_encoder = get_model_complexity_info(encoder, (3, size, size),
                                                                           as_strings=False,
                                                                           print_per_layer_stat=False,
                                                                           input_constructor=input_constructor,
                                                                           custom_modules_hooks=custom_module_mapping)
        encoder.use_encoder = True
        macs_encoder, params_encoder = get_model_complexity_info(encoder, (3, size, size), as_strings=False,
                                                                 print_per_layer_stat=False, verbose=False,
                                                                 input_constructor=input_constructor,
                                                                 custom_modules_hooks=custom_module_mapping)

        macs_full, params_full = get_model_complexity_info(student_model, (3, size, size), as_strings=False,
                                                           print_per_layer_stat=False, verbose=False,
                                                           input_constructor=input_constructor,
                                                           custom_modules_hooks=custom_module_mapping)


    ### Hotfix??? ####
    params_encoder = sum(p.numel() for p in encoder.encoder.parameters() if p.requires_grad)
    params_encoder += params_base_encoder
    ####################

    results['input_size'] = size
    results["jpeg_size"] = jpeg_size
    results["output_shape"] = [int(x) for x in encoder.output_shape]
    results["compression"] = np.prod(results["output_shape"]) / (size * size * 3)
    results['output_size'] = np.prod(results["output_shape"])*num_bits/8.0
    results["jpeg_compression"] = jpeg_size / (size * size * 3)
    results["macs_base_encoder"] = macs_base_encoder
    results["params_base_encoder"] = params_base_encoder
    results["macs_compressor"] = macs_encoder - macs_base_encoder
    results["params_compressor"] = params_encoder - params_base_encoder
    results["macs_decoder"] = macs_full - macs_encoder
    results["params_decoder"] = params_full - params_encoder

    if setting == 'Teacher':
        results["output_shape"] = [int(x) for x in encoder.original_output_shape]
        results["compression"] = np.prod(results["output_shape"]) / (size * size * 3)
        results["macs_compressor"] = 0.0
        results["params_compressor"] = 0.0

    del student_model
    del encoder

    return deepcopy(results)