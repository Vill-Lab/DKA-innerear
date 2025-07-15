from typing import Dict, List, Tuple

import multiprocessing as mp
import os
import tempfile
from enum import Enum
from functools import partial

import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from lighter.utils.model import adjust_prefix_and_load_state_dict
from tqdm import tqdm


def load_scan(path_dict):
    """
    Load and preprocess a CT scan from a file path or uploaded file.

    Args:
        path (str or UploadedFile): The file path or uploaded file object of the CT scan.

    Returns:
        dict: A dictionary containing the preprocessed CT scan image tensor with key "image".
              Returns None if the input path is None.
    """
    if path_dict is None:
        return None

    # Define the preprocessing transforms
    transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="SPL"),
            monai.transforms.Spacingd(keys=["image", "label"], pixdim=[1, 1, 1], mode="bilinear"),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        ]
    )

    res_dict = transforms(path_dict)
    return res_dict


class Model_type(Enum):
    ct_fm = "ct_fm"
    suprem = "suprem"


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_type=Model_type.ct_fm):
        super().__init__()
        if model_type == Model_type.ct_fm:
            self.model = adjust_prefix_and_load_state_dict(
                ckpt_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt",
                ckpt_to_model_prefix={"backbone.": ""},
                model=monai.networks.nets.segresnet_ds.SegResEncoder(
                    spatial_dims=3,
                    in_channels=1,
                    init_filters=32,
                    blocks_down=[1, 2, 2, 4, 4],
                    head_module=lambda x: x[-1],
                ),
            )

        elif model_type == Model_type.suprem:
            import sys

            sys.path.append("/home/suraj/Repositories/lighter-ct-fm")

            from models.backbones.unet3d import UNet3D
            from models.suprem import SuPreM_loader

            self.model = SuPreM_loader(
                model=UNet3D(n_class=10),
                ckpt_path="/mnt/data1/CT_FM/baselines/SuPreM_UNet/supervised_suprem_unet_2100.pth",
                decoder=False,
                encoder_only=True,
            )

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 3, 2)
        x = x.flip(2).flip(3)
        x = self.model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


def load_model(device="cuda", model_type=Model_type.ct_fm):
    model = EmbeddingModel(model_type)
    model.to(torch.device(device) if torch.cuda.is_available() else torch.device("cuda:0"))
    model.eval()
    return model


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


def search(query_scan, target_scan, query_point, patch_size, model_type=Model_type.ct_fm):
    # Crop and pad the reference patch around the selected point
    cropper = monai.transforms.Crop()
    padder = monai.transforms.SpatialPad(spatial_size=patch_size, mode="constant")
    slices = cropper.compute_slices(roi_center=query_point, roi_size=patch_size)
    query_patch = padder(cropper(query_scan, slices)).to("cuda:0")

    global model

    with torch.no_grad():
        if "model" not in globals():
            model = load_model(device="cuda:0", model_type=model_type)

        query_embedding = model(query_patch.unsqueeze(0))
        sim_fn = torch.nn.CosineSimilarity()

        def predictor(x):
            x = x.to("cuda:0")
            return sim_fn(model(x), query_embedding)

        target_scan = target_scan.to("cuda:0")
        splitter = monai.inferers.SlidingWindowSplitter(patch_size, 0.625)
        dataset = IterableDataset(splitter(target_scan.unsqueeze(0)))
        patch_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        similarity = []
        sim_heatmap = torch.zeros(target_scan.shape)
        for patch, location in patch_dataloader:
            sim = predictor(patch.squeeze(dim=1))
            for d, z, y, x in zip(sim, location[0], location[1], location[2]):
                similarity.append(
                    {
                        "sim": d.item(),
                        "location": (
                            z.item() + patch_size[0] // 2,
                            y.item() + patch_size[1] // 2,
                            x.item() + patch_size[2] // 2,
                        ),
                    }
                )
                sim_heatmap[
                    0,
                    z : z + patch_size[0],
                    y : y + patch_size[1],
                    x : x + patch_size[2],
                ] = d.item()

        max_sim = max(similarity, key=lambda x: x["sim"])
        sim_heatmap = (sim_heatmap - sim_heatmap.min()) / (sim_heatmap.max() - sim_heatmap.min())
        sim_heatmap = monai.transforms.GaussianSmooth(sigma=5.0)(sim_heatmap)

        return max_sim["location"], sim_heatmap


def get_query_point(label, centroid_label=51):
    # Get the centroid of the specified label in the image_dict using SimpleITK
    label = torch.where(label == centroid_label, 1, 0).numpy()[0]
    label_image = sitk.GetImageFromArray(label)
    label_shape_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics.Execute(label_image)
    centroid = label_shape_statistics.GetCentroid(1)
    centroid = label_image.TransformPhysicalPointToContinuousIndex(centroid)
    centroid = torch.tensor(centroid[::-1]).int()
    return centroid


def process_single_scan_pair(
    scan_pair: Tuple[Dict[str, str], Dict[str, str]],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    model_type=Model_type.ct_fm,
) -> float:
    """
    Process a single pair of scans and return the distance metric.

    Args:
        scan_pair: Tuple containing (query_scan_paths, target_scan_paths)
        patch_size: Size of the patch for searching

    Returns:
        float: Distance metric in mm
    """
    try:
        query_scan_paths, target_scan_paths = scan_pair

        # Load scans
        query_scan = load_scan(query_scan_paths)
        target_scan = load_scan(target_scan_paths)

        # Get query and target points
        query_point = get_query_point(query_scan["label"])
        target_point = get_query_point(target_scan["label"])

        # Perform search
        match_point, _ = search(
            query_scan["image"],
            target_scan["image"],
            query_point,
            patch_size,
            model_type=model_type,
        )

        # Calculate distance
        distance = torch.dist(torch.tensor(match_point, dtype=torch.float32), target_point.float())

        label_indices = torch.argwhere(target_scan["label"] == 51)[:, 1:]
        is_inside_label = torch.tensor(match_point) in label_indices
        return distance.item() / 10, is_inside_label  # Convert to mm
    except Exception as e:
        print(f"Error processing scan pair: {str(e)}")
        return None, None


def init_worker(model_type=Model_type.ct_fm):
    """Initialize worker process by loading the model into global scope"""
    global model
    model = load_model(device="cuda:0", model_type=model_type)


def process_batch_scans_parallel(
    scan_pairs: List[Tuple[Dict[str, str], Dict[str, str]]],
    num_processes: int = None,
    model_type=Model_type.ct_fm,
) -> List[float]:
    """
    Process multiple pairs of scans in parallel and calculate distances.

    Args:
        scan_pairs: List of tuples, each containing query and target scan path dictionaries
        num_processes: Number of processes to use. If None, uses cuda:0 count

    Returns:
        List of distances (in mm) for each pair
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Create a pool of workers with the model pre-loaded
    with mp.Pool(processes=num_processes, initializer=partial(init_worker, model_type=model_type)) as pool:
        # Process scan pairs in parallel with progress bar
        results = list(
            tqdm(
                pool.imap(partial(process_single_scan_pair, model_type=model_type), scan_pairs),
                total=len(scan_pairs),
                desc=f"Processing scan pairs using {num_processes} processes",
            )
        )

    print(results)
    return [r[0] for r in results], [r[1] for r in results]


import argparse

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process scan pairs and calculate semantic distances.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/data1/TotalSegmentator/v2/processed",
        help="Directory containing the data",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ct_fm",
        help="Model type to use for processing",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use for parallel processing",
    )

    args = parser.parse_args()

    model_type = Model_type[args.model_type]

    # Example usage with multiple scan pairs
    df = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    df = df.reset_index()
    df = df[df["study_type"].str.contains("thorax", na=False)]

    scan_pairs = []
    scan_id = df.iloc[0]["image_id"]
    ct = os.path.join(args.data_dir, scan_id, "ct.nii.gz")
    label = os.path.join(args.data_dir, scan_id, "label.nii.gz")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx == 0:
            continue
        scan_id2 = row["image_id"]
        ct2 = os.path.join(args.data_dir, scan_id2, "ct.nii.gz")
        label2 = os.path.join(args.data_dir, scan_id2, "label.nii.gz")
        scan_pairs.append(({"image": ct, "label": label}, {"image": ct2, "label": label2}))

    # test_scan_pairs = scan_pairs[:10]

    print(f"Processing {len(scan_pairs)} scan pairs using {args.num_processes} processes...")
    # Process scans in parallel
    distances, matches = process_batch_scans_parallel(scan_pairs, num_processes=args.num_processes, model_type=model_type)

    # Calculate statistics for valid distances
    valid_distances = [d for d in distances if d is not None]
    if valid_distances:
        mean_distance = np.mean(valid_distances)
        std_distance = np.std(valid_distances)
        print(f"\nResults:")
        print(f"Average distance: {mean_distance:.2f} ± {std_distance:.2f} cm")
        print(f"Successfully processed: {len(valid_distances)}/{len(distances)} pairs")

    valid_matches = [m for m in matches if m is not None]
    if valid_matches:
        print(f"Inside label: {sum(valid_matches)}/{len(valid_matches)}")

    # Save results
    df["semantic_distance"] = [None] + distances
    df["inside_label"] = [None] + matches

    df.to_csv(f"semantic_distances_{model_type}.csv", index=False)
