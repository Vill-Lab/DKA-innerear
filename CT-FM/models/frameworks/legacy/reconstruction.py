import torch
import torch.nn as nn


class Reconstruction(nn.Module):
    """
    Other frameworks (SimCLR, ConRecon, VicRegL) have `backbone` argument.
    This class does not implement any extra functionality - it just calls the backbone model.
    Exists only to follow the same pattern as other framework.
    """

    def __init__(self, backbone: nn.Module):
        """
        Args:
            backbone (nn.Module): The backbone model.
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, input):
        """Forward pass of the reconstruction model.

        Args:
            input: Input tensor to be passed through the backbone model.

        Returns:
            Output tensor from the backbone model.
        """
        return self.backbone(input)
