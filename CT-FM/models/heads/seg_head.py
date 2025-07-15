from __future__ import annotations

from typing import Union

import copy
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.segresnet_ds import SegResBlock
from monai.utils import UpsampleMode, has_option


class SegDecoder(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, out_channels: int, n_up=4):
        super().__init__()
        self.up_layers: nn.ModuleList = nn.ModuleList()
        self.mapping_layer = nn.Conv3d(embedding_dim, projection_dim, kernel_size=1)

        for i in range(n_up):
            projection_dim = projection_dim // 2
            kernel_size, stride = 3, 2

            level = nn.ModuleDict()
            level["upsample"] = UpSample(
                mode="deconv",
                spatial_dims=3,
                in_channels=2 * projection_dim,
                out_channels=projection_dim,
                kernel_size=kernel_size,
                scale_factor=stride,
                bias=False,
                align_corners=False,
            )
            blocks = [
                SegResBlock(
                    spatial_dims=3,
                    in_channels=projection_dim,
                    kernel_size=kernel_size,
                    norm="instance",
                    act="relu",
                )
            ]
            level["blocks"] = nn.Sequential(*blocks)
            self.up_layers.append(level)

        self.head = nn.Conv3d(
            in_channels=projection_dim,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        x = self.mapping_layer(x)
        for level in self.up_layers:
            x = level["upsample"](x)
            x = level["blocks"](x)
        x = self.head(x)
        return x
