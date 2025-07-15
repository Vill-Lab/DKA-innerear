import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead
from monai.data import decollate_batch


class ConRecon(nn.Module):
    """
    This class implements the ConRecon model, a variant of the SimCLR model + reconstruction.

    Attributes:
        backbone (nn.Module): The backbone network used in the ConRecon model.
        num_ftrs (int): The number of output features from the backbone network. Default is 32.
        out_dim (int): The dimension of the output representations. Default is 128.
        projection_head (SimCLRProjectionHead): The projection head used in the ConRecon model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int = 32,
        out_dim: int = 128,
        spatial_dims: int = 3,
    ):
        """
        Constructs the ConRecon model with a given backbone network, number of features, and output dimension.

        Args:
            backbone (nn.Module): The backbone network.
            num_ftrs (int, optional): The number of features. Default is 32.
            out_dim (int, optional): The output dimension. Default is 128.
            spatial_dims (int, optional): The number of spatial dimensions. Default is 3.
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs // 2, out_dim, batch_norm=False)
        if spatial_dims == 2:
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif spatial_dims == 3:
            self.average_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            raise ValueError("`spatial_dims` must be 2 or 3.")

    def forward(self, input):
        """
        Defines the computation performed at every call.

        Args:
            input (torch.Tensor or tuple or list): The input data. It can be a tensor, a tuple, or a list. Nested
                structures are also supported.

        Returns:
            dict: The output of the forward pass. The dictionary contains two keys: "con" and "recon".
                "con" corresponds to the contrastive representation and "recon" corresponds to the reconstruction map.
        """

        def tensor_forward(x):
            assert isinstance(x, torch.Tensor)
            embedding, outputs = self.backbone(x)
            embedding = self.average_pool(embedding).flatten(start_dim=1)
            embedding = self.projection_head(embedding)
            return embedding, outputs

        def sequence_forward(x):
            assert isinstance(x, tuple) or isinstance(x, list)
            embeddings = []
            maps = []
            for sample in x:
                if isinstance(sample, torch.Tensor):
                    embedding, outputs = tensor_forward(sample)
                    embeddings.append(embedding)
                    maps.append(outputs)

                elif isinstance(sample, tuple) or isinstance(sample, list):
                    embedding, outputs = sequence_forward(sample)
                    embeddings.append(embedding)
                    maps.append(outputs)

            return embeddings, maps

        if isinstance(input, torch.Tensor):
            embedding, map = tensor_forward(input)

        if isinstance(input, tuple) or isinstance(input, list):
            embedding, map = sequence_forward(input)

        return {"con": embedding, "recon": map}


if __name__ == "__main__":
    import sys

    sys.path.append("../backbones")
    from segresnetds_w_embedding import SegResNetDSwEmbedding

    # Create a ConRecon model

    backbone = SegResNetDSwEmbedding()
    model = ConRecon(backbone=backbone, num_ftrs=256, out_dim=128, spatial_dims=3)
    x = torch.rand(1, 1, 32, 32, 32)
    batch = (x, x)
    larger_batch = (batch, batch, batch, batch)
    out = model(larger_batch)
    print(len(out["con"]))
