import os
import pickle
from pathlib import Path

import medmnist
import monai
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import IterableDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

liver_df = pd.read_excel("/mnt/data1/datasets/MSD/2024.02.14.3D_MSD_Annotations.xlsx", sheet_name="01_liver")
colon_df = pd.read_excel("/mnt/data1/datasets/MSD/2024.02.14.3D_MSD_Annotations.xlsx", sheet_name="02_colon")
pancreas_df = pd.read_excel(
    "/mnt/data1/datasets/MSD/2024.02.14.3D_MSD_Annotations.xlsx",
    sheet_name="03_pancreas",
)
lung_df = pd.read_excel("/mnt/data1/datasets/MSD/2024.02.14.3D_MSD_Annotations.xlsx", sheet_name="04_lung")

df = pd.concat(
    [liver_df, colon_df, pancreas_df, lung_df],
    keys=["Liver", "Colon", "Pancreas", "Lung"],
)
df.reset_index(level=0, inplace=True)
df.rename(columns={"level_0": "Organ"}, inplace=True)

transforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        monai.transforms.EnsureTyped(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="SPL"),
        # monai.transforms.Orientationd(keys=["image"], axcodes="ras"),
        monai.transforms.Spacingd(keys=["image"], pixdim=[3, 1, 1], mode="bilinear"),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        monai.transforms.Lambda(func=lambda x: x["image"].as_tensor()),
    ]
)

import torch
from monai.networks.nets.segresnet_ds import SegResEncoder

weights = torch.load(
    "/mnt/data1/CT_FM/latest_fm_checkpoints/pretrained_segresnet.torch",
    map_location=torch.device("cpu"),
)
weights = {k.replace("encoder.", ""): v for k, v in weights.items()}

ctfm_model = SegResEncoder(
    blocks_down=(1, 2, 2, 4, 4),
    head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(x[-1], 1).flatten(
        start_dim=1
    ),  # Get only the last feature across block levels and average pool it.
)

ctfm_model.load_state_dict(weights, strict=False)  # Set strict to False as we load only the encoder
ctfm_model.eval()

data_dir = Path("/mnt/data1/datasets/MSD/combined")
data_list = list(data_dir.glob("*.nii.gz"))
feature_dict = {}

for data in tqdm(data_list):
    feature_dict[data] = {}
    img = transforms({"image": data})
    patch_size = (24, 128, 128)
    splitter = monai.inferers.SlidingWindowSplitter(patch_size, 0.625)
    dataset = IterableDataset(splitter(img.unsqueeze(0)))
    patch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    features = []
    with torch.no_grad():
        for batch, _ in patch_dataloader:
            features.extend([feat for feat in ctfm_model(batch.squeeze(1)).detach().cpu()])

    feature_dict[data]["features"] = features

pickle.dump(feature_dict, open("./MSD_features.pkl", "wb"))
