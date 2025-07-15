import torch
from monai.transforms import Compose, Transform


class BatchApply(Transform):
    def __init__(self, transforms, move_to_cpu=False):
        self.transforms = Compose(transforms)
        self.move_to_cpu = move_to_cpu

    def __call__(self, pred):
        assert pred.ndim in [
            4,
            5,
        ], f"Expected 4 (N,C,H,W) or 5 (N,C,H,W,D) dimensions, got {pred.ndim}"
        if self.move_to_cpu:
            pred = pred.cpu()

        pred = [self.transforms(p) for p in pred]
        pred = torch.stack(pred)
        return pred
