from typing import List

from lightly.loss import NegativeCosineSimilarity
from torch import nn


class SimSiamLoss(nn.Module):
    """
    SimSiamLoss:
    SimSiam Loss using Negative Cosine Similarity
    """

    def __init__(
        self,
    ):
        """
        Initialize an instance of the class.

        Args:
            dim (int, optional): The dimension along which the cosine similarity is computed. Defaults to 1.
            eps (float, optional): Small value to avoid division by zero. Defaults to 0.0001.
        """
        super().__init__()
        self.criterion = NegativeCosineSimilarity()

    def forward(self, out: List):
        """
        Forward pass through SimSiam Loss.

        Args:
            out (List[Tuple[torch.Tensor]]): List of tuples, each containing two tensors (z, p)

        Returns:
            float: SimSiam Loss value.
        """

        assert len(out) == 2, "Expecting two tuples as input"
        (z1, p1), (z2, p2) = out
        return self.criterion(z1, p2) / 2 + self.criterion(z2, p1) / 2
