import torch.nn as nn
import random
import torch
import numpy as np
import torch.distributed as dist
from torch import optim


# partially from https://github.com/XinhaoMei/WavCaps
def get_optimizer(params, lr, betas, eps, momentum, optimizer_name):
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            params, lr=lr, betas=betas, eps=eps
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            params, lr=lr, momentum=momentum
        )
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            params, lr=lr, betas=betas, eps=eps
        )
    else:
        raise ValueError("optimizer name is not correct")
    return optimizer


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, total_steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = total_steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def constant_lr(optimizer, base_lr, warmup_length): 
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def step_lr(optimizer, base_lr, warmup_length, adjust_steps, gamma):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if (step - warmup_length) > 0 and (step - warmup_length) % adjust_steps == 0:
                lr = optimizer.param_groups[0]["lr"] * gamma
            else:
                lr = optimizer.param_groups[0]["lr"]
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__": 
    naive_model = nn.Conv1d(in_channels=20, out_channels=448, kernel_size=7, padding=3)
    optimizer = get_optimizer(naive_model.parameters(),
                              lr=1e-4,
                              betas=[0.9, 0.999],
                              eps=1e-8,
                              momentum=0.9,
                              optimizer_name="adam")
    print(optimizer.param_groups)
    
    