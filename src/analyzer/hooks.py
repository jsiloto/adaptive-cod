import numpy as np




def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def usconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)

    in_channels = conv_module.in_channels_basic
    out_channels = conv_module.out_channels_basic

    if conv_module.slimmable_input:
        in_channels = int(round(conv_module.in_channels_basic * conv_module.width_mult, 0))
    if conv_module.slimmable_output:
        out_channels = int(round(conv_module.out_channels_basic * conv_module.width_mult, 0))

    print("conv_module.in_channels_basic", conv_module.in_channels_basic)
    print("conv_module.out_channels_basic", conv_module.out_channels_basic)
    print("in_channels", in_channels)
    print("out_channels", out_channels)

    groups = conv_module.groups
    print("groups", groups)

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)

    def empty_flops_counter_hook(module, input, output):
        module.__flops__ = 0

    # conv_module.conv.__flops_handle__.remove()
    if hasattr(conv_module, 'conv'):
        conv_module.conv.register_forward_hook(empty_flops_counter_hook)
    # if conv_module.depthwise:
    #     print(groups)
    #     print(filters_per_channel)
    #
    #     exit()
    # print(conv_module.__dict__)
    # print(conv_module.conv.__dict__)
    # exit()
    # print(conv_module.__flops__)
    # print(conv_module.conv.__flops__)
    # print(conv_module)

