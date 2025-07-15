import argparse
import os
import pickle
from pathlib import Path

import monai
import numpy as np
import torch
from monai.networks.nets.segresnet_ds import SegResEncoder
from tqdm import tqdm

from utils import IterableDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

TRANSFORMS = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        monai.transforms.EnsureTyped(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="SPL"),
        monai.transforms.Spacingd(keys=["image"], pixdim=[3, 1, 1], mode="bilinear"),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        monai.transforms.Lambda(func=lambda x: x["image"].as_tensor()),
    ]
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the image files")
parser.add_argument(
    "--extension",
    type=str,
    required=True,
    help='File extension to look for (e.g., "*.nrrd")',
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Output pickle file path to save features",
)
args = parser.parse_args()

# Load pretrained model
weights = torch.load(
    "/mnt/data1/CT_FM/latest_fm_checkpoints/pretrained_segresnet.torch",
    map_location=torch.device("cpu"),
)
weights = {k.replace("encoder.", ""): v for k, v in weights.items()}

ctfm_model = SegResEncoder(
    blocks_down=(1, 2, 2, 4, 4),
    head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(x[-1], 1).flatten(start_dim=1),
)
ctfm_model.load_state_dict(weights, strict=False)
ctfm_model.eval()

# Process data
data_dir = Path(args.data_dir)
data_list = list(data_dir.rglob(args.extension))
feature_dict = {}

for data in tqdm(data_list):
    feature_dict[data] = {}
    img = TRANSFORMS({"image": data})

    patch_size = (24, 128, 128)
    splitter = monai.inferers.SlidingWindowSplitter(patch_size, 0.0)
    dataset = IterableDataset(splitter(img.unsqueeze(0)))
    patch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    features = []
    with torch.no_grad():
        for batch, _ in patch_dataloader:
            features.extend([feat for feat in ctfm_model(batch.squeeze(1)).detach().cpu()])

    feature_dict[data]["features"] = features

# Save features
pickle.dump(feature_dict, open(args.output_file, "wb"))
