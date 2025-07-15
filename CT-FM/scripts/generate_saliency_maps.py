import gc
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import monai
import torch
import torch.nn as nn
from lighter.utils.model import adjust_prefix_and_load_state_dict
from monai.networks.nets import SegResNetDS
from monai.visualize import OcclusionSensitivity
from torch.nn.functional import cosine_similarity

# Data paths and setup
image_dir = Path("/mnt/data1/suraj/RadiomicsFoundationModel/LUNG1/NRRDs")
images = list(image_dir.rglob("*CT.nrrd"))

# Preprocessing transforms
keys = ["image", "label"]
image_key = "image"
transforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=keys, ensure_channel_first=True),
        monai.transforms.EnsureTyped(keys=keys),
        monai.transforms.Orientationd(keys=keys, axcodes="SPL"),
        monai.transforms.Spacingd(keys=keys, pixdim=[3, 3, 3], mode="bilinear"),
        monai.transforms.CropForegroundd(keys=keys, source_key=image_key),
        monai.transforms.ScaleIntensityRanged(keys=image_key, a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        monai.transforms.Lambda(func=lambda x: [x[k].as_tensor() for k in keys]),
    ]
)

# Process images
random.shuffle(images)
images = [transforms({"image": image, "label": str(image).replace("CT.nrrd", "masks/GTV-1.nrrd")}) for image in images[:10]]

print(images)


class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = adjust_prefix_and_load_state_dict(
            ckpt_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt",
            ckpt_to_model_prefix={"backbone.": "encoder"},
            model=SegResNetDS(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=[1, 2, 2, 4, 4],
                dsdepth=4,
            ),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.model.encoder(x)
        x.reverse()
        x = x.pop(0)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


# Setup model and occlusion sensitivity
encoder = EmbeddingModel().to("cuda:0")
occ_sens = OcclusionSensitivity(nn_module=encoder, n_batch=12, activate=False, mask_size=10, overlap=0.25)

# Generate distance maps
distance_maps = []
for i, x in enumerate(images):
    print(f"Processing image {i+1}/{len(images)}")
    x = x[0].unsqueeze(0).to("cuda:0")
    print(x.shape)
    occ_map, _ = occ_sens(x)

    encoder.eval()
    with torch.no_grad():
        base_embedding = encoder(x)
        distances = 1 - cosine_similarity(base_embedding.flatten().unsqueeze(0), occ_map.view(512, -1).t(), dim=1).squeeze()

        distances = distances.view(*occ_map.shape[2:])
        distance_maps.append(distances.cpu())

        torch.cuda.empty_cache()
        gc.collect()

# Save results
torch.save(
    {"distance_maps": distance_maps, "original_images": images},
    "lung1_saliency_data.torch",
)
