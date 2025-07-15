import torch
from torch import Tensor, nn


class IntraSampleNTXEntLoss(nn.Module):
    """
    This class implements the IntraSampleNTXEntLoss, a loss function used for contrastive learning.
    """

    def __init__(self, temperature: float = 0.1, permute=True, variance_weight=0.0):
        """
        Initialize an instance of the class.

        Args:
            temperature (float, optional): The temperature parameter for the instance. Defaults to 0.1.
            permute (bool): Whether to permute across all crops or just first crop. Defaults to True.
            variance_weight (float): Weight for variance regularization term. Defaults to 0.0.
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8
        self.permute = permute
        self.variance_weight = variance_weight

        if abs(self.temperature) < self.eps:
            raise ValueError(f"Illegal temperature: abs({self.temperature}) < 1e-8")

    def forward(self, input):
        """
        The forward pass of the IntraSampleNTXEntLoss.

        Args:
            input (list of list of Tensors): The input data in the following format:
                List [crops], List [views], Tensor [B, out_dim].
                Each batch index represents a different image.
        Returns:
            float: Contrastive Cross Entropy Loss value.
        """
        batch_size = input[0][0].shape[0]
        num_crops = len(input)
        device = input[0][0].device

        # Pre-normalize all tensors
        for i in range(num_crops):
            input[i][0] = nn.functional.normalize(input[i][0], dim=1)
            input[i][1] = nn.functional.normalize(input[i][1], dim=1)

        inv_loss = 0
        labels = torch.zeros(1, dtype=torch.long, device=device)

        crop_range = range(num_crops) if self.permute else range(1)
        for b in range(batch_size):
            per_sample_loss = 0
            for i in crop_range:
                # Get positives for current crop
                positive_0 = input[i][0][b : b + 1]
                positive_1 = input[i][1][b : b + 1]

                # Build negatives tensor directly
                negatives = []
                for j in range(num_crops):
                    if j != i:
                        negatives.extend([input[j][0][b], input[j][1][b]])

                if negatives:  # Only process if we have negatives
                    negatives = torch.stack(negatives)

                    # Compute similarities
                    sim_pos = torch.mm(positive_0, positive_1.t())
                    sim_neg = torch.mm(positive_0, negatives.t())

                    # Compute loss
                    logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
                    per_sample_loss += self.cross_entropy(logits, labels)

            inv_loss += per_sample_loss / len(crop_range)
        inv_loss /= batch_size

        if self.variance_weight > 0:
            # Compute variance loss only if needed
            positives_0 = torch.cat([input[i][0] for i in range(num_crops)], dim=0)
            positives_1 = torch.cat([input[i][1] for i in range(num_crops)], dim=0)
            var_loss = 0.5 * (variance_loss(positives_0, self.eps) + variance_loss(positives_1, self.eps))
            return (1 - self.variance_weight) * inv_loss + self.variance_weight * var_loss

        return inv_loss


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x: Tensor with shape (batch_size, ..., dim).
        eps: Epsilon for numerical stability.

    Returns:
        The computed VICReg variance loss.
    """
    return torch.mean(torch.nn.functional.relu(1.0 - torch.sqrt(x.var(dim=0) + eps)))
