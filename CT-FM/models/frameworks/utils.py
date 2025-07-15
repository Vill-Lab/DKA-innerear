import torch
from torch import nn


def tensor_forward(obj, x):
    assert isinstance(x, torch.Tensor)
    x = obj.backbone(x)
    x = obj.average_pool(x).flatten(start_dim=1)
    x = obj.projection_head(x)
    return x


def sequence_forward(obj, x):
    assert isinstance(x, tuple) or isinstance(x, list)
    out = []
    for sample in x:
        if isinstance(sample, torch.Tensor):
            out.append(tensor_forward(obj, sample))
        elif isinstance(sample, tuple) or isinstance(sample, list):
            out.append(sequence_forward(obj, sample))
    return out


def AdaptiveAvgPool(spatial_dims):
    if spatial_dims == 2:
        return nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif spatial_dims == 3:
        return nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    else:
        raise ValueError("`spatial_dims` must be 2 or 3.")
