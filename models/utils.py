import numpy as np
import torch
import random
import os
import scipy

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_init(conv):
    """
    Initialize convolutional layer weights using Kaiming normal initialization and set biases to zero.

    Parameters
    ----------
    conv : torch.nn.Conv2d or torch.nn.Conv1d
        The convolutional layer to be initialized.
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """
    Initialize batch normalization layer weights to a specified scale and set biases to zero.

    Parameters
    ----------
    bn : torch.nn.BatchNorm2d or torch.nn.BatchNorm1d
        The batch normalization layer to be initialized.
    scale : float
        The scale value to initialize the weights of the batch normalization layer.
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


from typing import Optional

def global_mean_pooling(x, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Perform global mean pooling on the input tensor, considering the provided mask.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, feature_dim).
    mask : torch.Tensor
        Mask tensor of shape (batch_size, seq_len) where 1 indicates valid data and 0 indicates padding.

    Returns
    -------
    torch.Tensor
        Pooled tensor of shape (batch_size, feature_dim).
    """
    if mask is None:
        mask = (x != 0).float().unsqueeze(-1).to(dtype=torch.float32)
    else:
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

    sum_len = mask.sum(dim=1).clamp_min(1.0)  # Avoid division by zero
    pooled_x = (x * mask).sum(dim=1) / sum_len  # [batch_size, feature_dim]

    return pooled_x

import math

def lr_warmup_function(epochs:int, len_train: int):
    """
    Create a learning-rate schedule with linear warmup and cosine decay.

    The schedule assumes one scheduler step per training batch. Warmup lasts
    for 5 percent of the total training steps, after which the learning-rate
    multiplier follows a cosine decay.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    len_train : int
        Number of training batches per epoch.

    Returns
    -------
    callable
        A function that accepts the current step and returns the
        corresponding learning-rate multiplier.
    """
    steps = epochs * len_train
    warmup = int(0.05 * steps)

    def lr_lambda(step):
        if step < warmup: 
            return step / max(1, warmup)
        p = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    return lr_lambda