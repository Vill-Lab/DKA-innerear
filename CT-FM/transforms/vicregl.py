from typing import Dict, List, Tuple

from dataclasses import dataclass

import monai
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop, Flip, RandSpatialCrop, Transform
from torch import nn


@dataclass
class Location:
    # The depth index of the top-left corner of the crop.
    front: float
    # The row index of the top-left corner of the crop.
    top: float
    # The column index of the top-left corner of the crop.
    left: float
    # The depth of the crop.
    depth: float
    # The height of the crop.
    height: float
    # The width of the crop.
    width: float
    # The depth of the original image.
    image_depth: float
    # The height of the original image.
    image_height: float
    # The width of the original image.
    image_width: float
    # Whether to flip the image along depth axis.
    depth_flip: bool = False
    # Whether to flip the image horizontally.
    horizontal_flip: bool = False
    # Whether to flip the image vertically.
    vertical_flip: bool = False


class RandSpatialCropWithLocation(RandSpatialCrop):
    """
    Do a random spatial crop and return both the resulting image and the location.
    """

    def __init__(self, roi_size) -> None:
        super().__init__(roi_size=roi_size, random_size=False)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Location]:
        """
        Args:
            img (torch.Tensor): Image to be cropped.
        Returns:
            torch.Tensor: Randomly cropped image
            Location: Location object containing crop parameters
        """
        # Calculates the self._slices

        cropped_img = super().__call__(img)

        front_slice, top_slice, left_slice = self._slices
        front, top, left = front_slice.start, top_slice.start, left_slice.start
        depth, height, width = (
            front_slice.stop - front_slice.start,
            top_slice.stop - top_slice.start,
            left_slice.stop - left_slice.start,
        )
        image_depth, image_height, image_width = img.shape[-3:]
        location = Location(
            front=front,
            top=top,
            left=left,
            depth=depth,
            height=height,
            width=width,
            image_depth=image_depth,
            image_height=image_height,
            image_width=image_width,
        )

        return cropped_img, location


class RandomResizedCropAndFlip3D(Transform):
    """Randomly flip and crop a 3D image.
    A PyTorch module that applies random spatial cropping, depth, horizontal and vertical flipping to a 3D image,
    and returns the transformed image and a grid tensor used to map the image back to the
    original image space in an NxNxN grid.
    Args:
        grid_size (List[int]):
            The number of grid cells in the output grid tensor in ZXY shape.
        roi_size:
            The size (in pixels) of the random spatial crops.
        prob:
            The probability of applying the random spatial crop.
        depth_flip_prob:
            The probability of applying depth-wise flipping transformation.
        horizontal_flip_prob:
            The probability of applying horizontal flipping transformation.
        vertical_flip_prob:
            The probability of applying vertical flipping transformation.
    """

    def __init__(
        self,
        roi_size: List[int],
        grid_size: List[int],
        depth_flip_prob: float = 0,
        horizontal_flip_prob: float = 0,
        vertical_flip_prob: float = 0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.roi_size = roi_size
        self.spatial_crop = RandSpatialCropWithLocation(roi_size=self.roi_size)
        self.depth_flip = Flip(spatial_axis=0)
        self.vertical_flip = Flip(spatial_axis=1)
        self.horizontal_flip = Flip(spatial_axis=2)
        self.depth_flip_prob = depth_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.horizontal_flip_prob = horizontal_flip_prob

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies random cropping and depth, horizontal, and vertical flipping to a 3D image, and returns
        the transformed image and a grid tensor used to map the image back to the original image
        space in an NxNxN grid.
        Args:
            img: The input torch.Tensor image.
        Returns:
            A tuple containing the transformed torch.Tensor image and the grid tensor.
        """

        img, location = self.spatial_crop(img)

        # Create grid
        grid = self.location_to_NxNxN_grid(location=location)

        # Depth flip
        location.depth_flip = False
        if np.random.rand() < self.depth_flip_prob:
            img = self.depth_flip(img)
            location.depth_flip = True

        # Vertical flip
        location.vertical_flip = False
        if np.random.rand() < self.vertical_flip_prob:
            img = self.vertical_flip(img)
            location.vertical_flip = True

        # Horizontal flip
        location.horizontal_flip = False
        if np.random.rand() < self.horizontal_flip_prob:
            img = self.horizontal_flip(img)
            location.horizontal_flip = True

        return {"image": img, "grid": grid}

    def location_to_NxNxN_grid(self, location: Location) -> torch.Tensor:
        """Create grid from location object.
        Create a grid tensor with grid_size rows, grid_size columns, and grid_size depth, where each cell represents a
        region of the original image. The grid is used to map the cropped and transformed image back to the
        original image space.
        Args:
            location: An instance of the Location class, containing the location and size of the
                transformed image in the original image space.
        Returns:
            A grid tensor of shape (grid_size, grid_size, grid_size, 3), where the last dimension represents the
            (x, y, z) coordinate of the center of each cell in the original image space.
        """

        cell_depth = location.depth / self.grid_size[0]
        cell_height = location.height / self.grid_size[1]
        cell_width = location.width / self.grid_size[2]
        z = torch.linspace(location.front, location.front + location.depth, self.grid_size[0]) + (cell_depth / 2)
        y = torch.linspace(location.top, location.top + location.height, self.grid_size[1]) + (cell_height / 2)
        x = torch.linspace(location.left, location.left + location.width, self.grid_size[2]) + (cell_width / 2)
        if location.depth_flip:
            z = torch.flip(z, dims=[0])
        if location.vertical_flip:
            y = torch.flip(y, dims=[0])
        if location.horizontal_flip:
            x = torch.flip(x, dims=[0])
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([grid_z, grid_y, grid_x], dim=-1)
