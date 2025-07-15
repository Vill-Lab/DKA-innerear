import torch
from torch import nn


class IntraSampleWrapper(nn.Module):
    """
    This class implements a wrapper for a loss function used for contrastive learning.
    """

    def __init__(self, loss):
        """
        Initialize an instance of the class.

        Args:
            criterion (nn.Module): The loss criterion to be wrapped.
        """
        super().__init__()
        self.loss = loss

    def forward(self, input):
        """
        The forward pass of the IntraSampleWrapper. This wrapper converts
        a list of list of batched tensors where the first list is of the number of crops and the second
        is of views to a format where the number of crops is the batch dimension.

        Args:
            input (list of list of Tensor or Tensor tuple): The input data in the following format:
                List [crops], List [views], Tensor or Tensor tuple with dims [B, out_dim].
                Each batch index represents a different image.
        Returns:
            float: Contrastive loss value.
        """
        n_views = len(input[0])

        if isinstance(input[0][0], tuple):
            batch = torch.stack([torch.stack([torch.stack(crop[view]) for view in range(n_views)]) for crop in input])
        else:
            batch = torch.stack([torch.stack([crop[view] for view in range(n_views)]) for crop in input])

        if batch.ndim == 4:
            batch = batch.permute(2, 1, 0, 3)
            batch = [[view for view in batch_el] for batch_el in batch]
        elif batch.ndim == 5:
            batch = batch.permute(3, 1, 2, 0, 4)
            batch = [[[tensor for tensor in view] for view in batch_el] for batch_el in batch]

        loss = sum(self.loss(batch_item) for batch_item in batch) / len(batch)
        return loss
