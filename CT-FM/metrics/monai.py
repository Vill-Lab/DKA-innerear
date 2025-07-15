from monai.metrics import DiceHelper
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class DiceScore(Metric):
    """
    Computes the Generalized Dice Score for segmentation tasks.

    Args:
        include_background (bool): Whether to include the background class in the computation.
        per_class (bool): Whether to compute the score per class or as a single value.
    """

    def __init__(self, include_background: bool = False, per_class: bool = False):
        super().__init__()
        reduction = "mean_batch" if per_class else "mean"
        self.metric = DiceHelper(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=False,
            ignore_empty=True,
        )
        self.add_state("dice", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the state with predictions and targets.

        Args:
            preds (Tensor): Predicted tensor.
            target (Tensor): Ground truth tensor.
        """
        if target.ndim == 4:
            target = target.unsqueeze(1)
        self.dice.append(self.metric(preds, target))

    def compute(self) -> Tensor:
        """
        Compute the final Generalized Dice Score.

        Returns:
            Tensor: The computed Generalized Dice Score.
        """
        return dim_zero_cat(self.dice)
