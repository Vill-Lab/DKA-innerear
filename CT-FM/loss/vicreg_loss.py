from typing import List

from lightly.loss import VICRegLoss as lightly_VICRegLoss


class VICRegLoss(lightly_VICRegLoss):
    """
    VICRegLoss:
    Variance-Invariance-Covariance Regularization Loss
    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps=0.0001,
    ):
        """
        Initialize an instance of the class.

        Args:
            lambda_param (float, optional): The lambda parameter for the instance. Defaults to 25.0.
            mu_param (float, optional): The mu parameter for the instance. Defaults to 25.0.
            nu_param (float, optional): The nu parameter for the instance. Defaults to 1.0.
            gather_distributed (bool, optional): Whether to gather distributed data. Defaults to False.
            eps (float, optional): Small value for numerical stability. Defaults to 0.0001.
        """
        super().__init__(lambda_param, mu_param, nu_param, gather_distributed, eps)

    def forward(self, out: List):
        """
        Forward pass through VICReg Loss.

        Args:
            out (List[torch.Tensor]): List of tensors

        Returns:
            float: VICReg Loss value.
        """
        return super().forward(*out)
