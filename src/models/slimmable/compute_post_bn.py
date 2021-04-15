import torch
from torch import nn
import time
from tqdm import tqdm

import torch.nn.functional as F
# from slimmable_ops import *

def adjust_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        # Removing the stats computed using exponential running average
        # and resetting count
        try:
            module.reset_running_stats()
        except:
            pass

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum
        
        # Switching to cumulutive running average
        module.momentum = 0.1
        
        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats
        
        module.training = True
        # module.affine = True
        module.track_running_stats = True
            
def restore_original_settings_of_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        
        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats

def adjust_momentum(module, t):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 1 / (t+1)

def ComputeBN(net, postloader, num_batch=1000):
    print("Computing Batch Norm")
    start = time.time()
    net.train()
    net.apply(adjust_bn_layers)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(postloader):
            net.apply(lambda m: adjust_momentum(m, batch_idx))
            inputs = [i.cuda() for i in inputs[0:1]]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets[0:1]]
            _ = net(inputs, targets)
            if not batch_idx < num_batch:
                break

    net.apply(restore_original_settings_of_bn_layers)
    net.eval()
    elapsed = time.time() - start
    print("Batch Norm Computed {}".format(elapsed))
    return net