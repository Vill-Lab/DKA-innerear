from lightly.models.modules.heads import (
    BarlowTwinsProjectionHead,
    VicRegLLocalProjectionHead,
)
from torch import nn

from ..utils import AdaptiveAvgPool


class VICRegL(nn.Module):
    def __init__(
        self,
        backbone,
        num_ftrs,
        global_projection_ftrs=(2048, 2048),
        local_projection_ftrs=(128, 128),
        spatial_dims=3,
    ):
        super().__init__()
        self.backbone = backbone
        self.global_projection_head = BarlowTwinsProjectionHead(num_ftrs, *global_projection_ftrs)
        self.local_projection_head = VicRegLLocalProjectionHead(num_ftrs, *local_projection_ftrs)
        self.average_pool = AdaptiveAvgPool(spatial_dims)

    def forward(self, input):
        high_resolution_crops = input["high_resolution_crops"]
        low_resolution_crops = input["low_resolution_crops"]

        global_views_features = [self.subforward(data["image"]) for data in high_resolution_crops]
        global_grids = [data["grid"] for data in high_resolution_crops]

        local_views_features = [self.subforward(data["image"]) for data in low_resolution_crops]
        local_grids = [data["grid"] for data in low_resolution_crops]

        return global_views_features, global_grids, local_views_features, local_grids

    def subforward(self, x):
        x = self.backbone(x)
        # Global features
        y_global = self.average_pool(x).flatten(start_dim=1)
        z_global = self.global_projection_head(y_global)

        # Local features
        dims_after_batch_and_channel = list(range(2, x.ndim))
        y_local = x.permute(0, *dims_after_batch_and_channel, 1)
        z_local = self.local_projection_head(y_local)

        return z_global, z_local
