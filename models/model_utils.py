import torch
import random
import numpy as np
import time
import sys
import logging
import os
from torch.nn import init
from torch import nn
from .linear_weightdrop import WeightDropLinear

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, WeightDropLinear):
            _init(m.weight_raw)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

def set_all_random_seed(SEED):
    # to reproduce the results
    random.seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def save_checkpoint(epoch, model, optimizer, model_kwargs, ckpt_dir, file_prefix = 'best'):
    model_ckpt = os.path.join(ckpt_dir, file_prefix + '_model.pt')
    model_state = {
        'model_kwargs': model_kwargs,
        'model_state' : model.state_dict()
    }
    time.sleep(10)
    torch.save(model_state, model_ckpt)

    if optimizer is not None:
        optim_ckpt = os.path.join(ckpt_dir, file_prefix + '_optimizer.pt')

        optim_state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(optim_state, optim_ckpt)

def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


if __name__ == '__main__':
    import torch
    bbox = image_box = torch.tensor([0.0,0.0,1.0,1.0])
    print(bbox.size())
    image_box = image_box.repeat(32, 16, 1, 1)
    print(image_box.size())