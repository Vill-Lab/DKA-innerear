import torch
import torch.nn as nn

from ..utils import apply_fns


class TrunkHeadWrapper(nn.Module):
    def __init__(self, trunk, head, pre_func=None):
        """

        Args:
            trunk (nn.Module): The base model to be used for feature extraction.
            head (list): A list of integers representing the number of neurons in each head layer.
            pre_func (nn.Module): A function to apply to the trunk output before the head layers.
        """
        super().__init__()
        self.trunk = trunk
        self.head = self._init_head(head)
        self.pre_func = pre_func

    def forward(self, x):
        x = self.trunk(x)

        if self.pre_func is not None:
            x = apply_fns(x, self.pre_func)

        if self.head is not None:
            x = self.head(x)

        return x

    def _init_head(self, head):
        if isinstance(head, list):
            return nn.Sequential(*head)
        else:
            return head
