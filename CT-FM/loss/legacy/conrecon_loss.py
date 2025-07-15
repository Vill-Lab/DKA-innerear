from torch import nn


class ReconLossNestedWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):
        loss = 0
        for input_samples, target_samples in zip(input, target):
            for input_duplicate, target_duplicate in zip(input_samples, target_samples):
                loss += self.loss(input_duplicate, target_duplicate)

        # Average the loss
        loss = loss / (len(input) * len(input[0]))
        return loss


class ConReconLoss(nn.Module):
    """
    This class implements the ConReconLoss, a loss function used for contrastive learning + reconstruction
    """

    def __init__(
        self,
        contrastive_loss: nn.Module,
        reconstruction_loss: nn.Module,
        alpha: float,
        beta: float,
    ):
        """
        Initialize an instance of the class.

        Args:
        """
        super().__init__()
        self.con_loss = contrastive_loss
        self.recon_loss = ReconLossNestedWrapper(reconstruction_loss)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        """
        The forward pass of the ConReconLoss.
        """
        if "con" not in input and "recon" not in input:
            raise ValueError("input must contain both contrastive and reconstruction data under 'con' and 'recon' keys.")
        con_loss = self.con_loss(input["con"])
        recon_loss = self.recon_loss(input["recon"], target)
        total_loss = self.alpha * con_loss + self.beta * recon_loss
        return {"con": con_loss, "recon": recon_loss, "total": total_loss}
