import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots_vals'))):
    def __call__(self, t):
        knots = [x[0] for x in self.knots_vals]
        vals = [x[1] for x in self.knots_vals]
        return np.interp([t], knots, vals)[0]


# Swish and MemoryEfficientSwish: Two implementations of the method
# round_filters and round_repeats:
#     Functions to calculate params for scaling model width and depth ! ! !
# get_width_and_height_from_size and calculate_output_image_size
# drop_connect: A structural design
# get_same_padding_conv2d:
#     Conv2dDynamicSamePadding
#     Conv2dStaticSamePadding
# get_same_padding_maxPool2d:
#     MaxPool2dDynamicSamePadding
#     MaxPool2dStaticSamePadding
#     It's an additional function, not used in EfficientNet,
#     but can be used in other model (such as EfficientDet).
# Identity: An implementation of identical mapping


# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LossTracker():
    def __init__(self, reduce_tensor):
        super().__init__()

        self.reduce_tensor = reduce_tensor

        self.total_loss = None
        self.loss_steps = 0
        self.nan_steps = 0

    def update(self, loss):
        # clone probably not needed
        loss = self.reduce_tensor(loss.detach().clone())
        is_nan = torch.isnan(loss).any()
        if is_nan:
            self.nan_steps += 1
        else:
            if self.total_loss is None:
                self.total_loss = loss
            else:
                self.total_loss += loss
            self.loss_steps += 1
        return is_nan

    def query_reset(self):
        if self.total_loss is None:
            avg_loss = None
        else:
            avg_loss = self.total_loss.item() / self.loss_steps
        self.total_loss = None
        self.loss_steps = 0
        self.nan_steps = 0

        return avg_loss, self.nan_steps


class CosineDecay():
    def __init__(self, start_x, start_y, end_x, end_y):
        super().__init__()

        self.scale = (start_y - end_y) / 2.0
        self.shift = end_y + self.scale
        self.scale_inner = math.pi / (end_x - start_x)
        self.shift_inner = -start_x

    def __call__(self, x):
        return math.cos((x + self.shift_inner) *
                        self.scale_inner) * self.scale + self.shift


class LRSched():
    def __init__(self,
                 lr_max,
                 epochs,
                 start_div_factor=4.0,
                 pct_start=0.1,
                 final_div_factor=None,
                 offset=0):
        start_lr = lr_max / start_div_factor
        if final_div_factor is None:
            final_lr = 0.0
        else:
            final_lr = lr_max / final_div_factor
        self.decay_start_epoch = int(pct_start * epochs)

        self.linear_part = PiecewiseLinear([(0, start_lr),
                                            (self.decay_start_epoch, lr_max)])
        if epochs - self.decay_start_epoch > 0:
            self.cos_part = CosineDecay(self.decay_start_epoch, lr_max, epochs,
                                        final_lr)
        self.offset = offset

    def __call__(self, epoch):
        epoch = epoch - self.offset
        if epoch <= self.decay_start_epoch:
            return self.linear_part(epoch)
        else:
            return self.cos_part(epoch)


class EMATracker():
    def __init__(self, alpha, start_value):
        self.alpha = alpha
        self.x = start_value

    def update(self, value):
        self.x = self.x * self.alpha + (1 - self.alpha) * value


def recursive_param_print(module, memo=None, value='', name="net"):
    if memo is None:
        memo = set()

    total = 0
    to_print = ""

    if module not in memo:
        memo.add(module)

        for m_name, c_module in module.named_children():
            child_total, child_to_print = recursive_param_print(c_module,
                                                                memo,
                                                                "  " + value,
                                                                name=m_name)

            total += child_total
            to_print += child_to_print

        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                this_param_total = param.numel()
                total += this_param_total
                to_print += "  " + value + p_name + " (param) : {}\n".format(
                    this_param_total)

        if total != 0:
            to_print = value + name + ": {}\n".format(total) + to_print

    return total, to_print


if __name__ == "__main__":
    epochs = 100
    sched = LRSched(100, epochs)

    import matplotlib.pyplot as plt

    x = []
    y = []

    for i in range(epochs):
        x.append(i)
        y.append(sched(i))

    plt.plot(x, y)
    plt.show()
