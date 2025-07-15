from typing import Any, Callable, Dict, List, Optional, Tuple

from copy import deepcopy

import torch
from lighter.utils.misc import ensure_list
from monai.transforms import MapTransform, RandScaleCrop, Resize, Transform


class Duplicate:
    """Duplicate an input and apply two different transforms. Used for SimCLR primarily."""

    def __init__(
        self,
        transforms1: Optional[Callable] = None,
        transforms2: Optional[Callable] = None,
    ):
        """Duplicates an input and applies the given transformations to each copy separately.
        Args:
            transforms1 (Optional[Callable], optional): _description_. Defaults to None.
            transforms2 (Optional[Callable], optional): _description_. Defaults to None.
        """
        # Wrapped into a list if it isn't one already to allow both a
        # list of transforms as well as `torchvision.transform.Compose` transforms.
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of two tensors.
        """
        out1, out2 = input, deepcopy(input)
        if self.transforms1 is not None:
            out1 = self.transforms1(out1)
        if self.transforms2 is not None:
            out2 = self.transforms2(out2)
        return (out1, out2)


class MultiCrop:
    """Multi-Crop augmentation."""

    def __init__(
        self,
        high_resolution_transforms: List[Callable],
        low_resolution_transforms: Optional[List[Callable]] = None,
    ):
        self.high_resolution_transforms = ensure_list(high_resolution_transforms)
        self.low_resolution_transforms = ensure_list(low_resolution_transforms)

    def __call__(self, input):
        high_resolution_crops = [transform(input) for transform in self.high_resolution_transforms]
        low_resolution_crops = [transform(input) for transform in self.low_resolution_transforms]
        return {
            "high_resolution_crops": high_resolution_crops,
            "low_resolution_crops": low_resolution_crops,
        }


class RandomResizedCrop(Transform):
    """
    Combines monai's random spatial crop followed by resize to the desired size.

    Modification:
    1. The spatial crop is done with same dimensions for all the axes
    2. Handles cases where the image_size is less than the crop_size by choosing
        the smallest dimension as the random scale.

    """

    def __init__(
        self,
        prob: float = 1,
        size: List[int] = [50, 50, 50],
        scale: List[float] = [0.5, 1.0],
    ):
        """
        Args:
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        """
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.size = size

    def __call__(self, image):
        if torch.rand(1) < self.prob:
            random_scale = torch.empty(1).uniform_(*self.scale).item()
            rand_cropper = RandScaleCrop(random_scale, random_size=False)
            resizer = Resize(self.size, mode="trilinear")
            image = rand_cropper(image)
            image = resizer(image)
        return image


class DictifyTransform(MapTransform):
    """
    Dict version of RandomResizedCrop.
    """

    def __init__(self, transform: Callable, keys: List[str], allow_missing_keys: bool = False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.transform = transform

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
            else:
                if not self.allow_missing_keys:
                    raise ValueError(f"Key {key} not found in the input dictionary.")
        return data
